
%%writefile gemm.cu

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error '%s' at %s:%d\n",                  \
                         cudaGetErrorString(err__), __FILE__, __LINE__);        \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st__ = (call);                                           \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                    \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st__,       \
                         __FILE__, __LINE__);                                   \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

////////////////////////////////////////////////////////////////////////////////
// Custom batched GEMM
//
// C[b] = A[b] * B[b]
//
// Matrices are N x N, column-major (N <= 32), one thread block per matrix.
////////////////////////////////////////////////////////////////////////////////
__global__ void batch_gemm_kernel(int n, int batch_size,
                                  const float *A,
                                  const float *B,
                                  float *C)
{
    // 1 block per matrix in the batch (uniform per block, so the early
    // return cannot cause a divergent __syncthreads)
    int matrix_idx = blockIdx.x;
    if (matrix_idx >= batch_size)
        return;

    // threadIdx.x -> row, threadIdx.y -> col.
    // Consecutive threads (x) then write consecutive addresses of C
    // (coalesced) and read consecutive words of shared_A (no bank conflicts);
    // shared_B accesses are broadcasts within a column.
    int row = threadIdx.x;
    int col = threadIdx.y;

    __shared__ float shared_A[32 * 32];
    __shared__ float shared_B[32 * 32];

    // size_t: avoid int overflow for large batches
    const size_t offset = (size_t)matrix_idx * n * n;
    const float *A_b = A + offset;
    const float *B_b = B + offset;
    float *C_b = C + offset;

    // Cooperative, coalesced load of both matrices into shared memory
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;

    for (int i = tid; i < n * n; i += total_threads)
    {
        shared_A[i] = A_b[i];
        shared_B[i] = B_b[i];
    }

    __syncthreads();

    if (row >= n || col >= n)
        return;

    // C(row,col) = sum_k A(row,k) * B(k,col)
    float sum = 0.0f;
    for (int k = 0; k < n; ++k)
    {
        sum += shared_A[k * n + row] * shared_B[col * n + k];
    }

    C_b[col * n + row] = sum;
}

////////////////////////////////////////////////////////////////////////////////
// CPU reference implementation (accumulates in double for a trustworthy
// reference)
void batched_gemm_cpu(int n, int batch_size,
                      const float *A,
                      const float *B,
                      float *C)
{
    const size_t matrix_size = (size_t)n * n;

    for (int b = 0; b < batch_size; ++b)
    {
        const size_t base = (size_t)b * matrix_size;
        for (int col = 0; col < n; ++col)
        {
            for (int row = 0; row < n; ++row)
            {
                double sum = 0.0;
                for (int k = 0; k < n; ++k)
                {
                    sum += (double)A[base + (size_t)k * n + row] *
                           (double)B[base + (size_t)col * n + k];
                }
                C[base + (size_t)col * n + row] = (float)sum;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Forward error for one matrix
double forward_error(int n,
                     const float *result,
                     const float *reference)
{
    double diff2 = 0.0;
    double ref2 = 0.0;
    const size_t size = (size_t)n * n;

    for (size_t i = 0; i < size; ++i)
    {
        const double diff = (double)result[i] - (double)reference[i];
        diff2 += diff * diff;
        ref2 += (double)reference[i] * reference[i];
    }

    const double diff_norm = std::sqrt(diff2);
    const double ref_norm = std::sqrt(ref2);

    if (ref_norm > 0.0)
        return diff_norm / (n * ref_norm);

    return (diff_norm == 0.0) ? 0.0 : INFINITY;
}

////////////////////////////////////////////////////////////////////////////////
// Maximum forward error over the batch
double max_batch_forward_error(int n,
                               int batch_size,
                               const float *result,
                               const float *reference)
{
    const size_t matrix_size = (size_t)n * n;
    double max_error = 0.0;

    for (int b = 0; b < batch_size; ++b)
    {
        const size_t base = (size_t)b * matrix_size;
        const double e = forward_error(n, result + base, reference + base);
        max_error = std::max(max_error, e);
    }

    return max_error;
}

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    const int n = (argc > 1) ? std::atoi(argv[1]) : 16;
    const int batch_size = (argc > 2) ? std::atoi(argv[2]) : 1000;
    const int repeats = 100;

    if (n <= 0 || n > 32 || batch_size <= 0)
    {
        std::fprintf(stderr, "Usage: %s matrix_size(1..32) batch_size\n", argv[0]);
        return EXIT_FAILURE;
    }

    const size_t matrix_size = (size_t)n * n;
    const size_t total = matrix_size * batch_size;
    const size_t bytes = total * sizeof(float);

    // --------------------------------------------------------
    // Initialize input matrices
    // --------------------------------------------------------
    std::vector<float> h_A(total);
    std::vector<float> h_B(total);
    std::vector<float> h_ref(total);
    std::vector<float> h_custom(total);
    std::vector<float> h_cublas(total);

    for (size_t i = 0; i < total; ++i)
    {
        h_A[i] = std::sin(0.013f * (float)(i + 1));
        h_B[i] = std::cos(0.017f * (float)(i + 1));
    }

    // ref. CPU GEMM
    batched_gemm_cpu(n, batch_size, h_A.data(), h_B.data(), h_ref.data());

    // --------------------------------------------------------
    // Allocate GPU memory
    // --------------------------------------------------------
    float *d_A = nullptr, *d_B = nullptr, *d_C_custom = nullptr, *d_C_cublas = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C_custom, bytes));
    CUDA_CHECK(cudaMalloc(&d_C_cublas, bytes));

    // --------------------------------------------------------
    // Copy A and B to GPU
    // --------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // --------------------------------------------------------
    // GPU information
    // --------------------------------------------------------
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::printf("GPU: %s (compute capability %d.%d)\n", prop.name, prop.major, prop.minor);
    std::printf("Matrix size: %d x %d, batch size: %d\n", n, n, batch_size);
    std::printf("FP32, column-major matrices, repeats: %d\n\n", repeats);

    // --------------------------------------------------------
    // CUDA events
    // --------------------------------------------------------
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --------------------------------------------------------
    // Custom kernel configuration: 1 block per matrix, n x n threads
    // --------------------------------------------------------
    dim3 grid(batch_size);
    dim3 block(n, n);

    // --------------------------------------------------------
    // Warmup run
    // --------------------------------------------------------
    batch_gemm_kernel<<<grid, block>>>(n, batch_size, d_A, d_B, d_C_custom);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timing custom kernel
    CUDA_CHECK(cudaEventRecord(start, 0));
    for (int r = 0; r < repeats; ++r)
    {
        batch_gemm_kernel<<<grid, block>>>(n, batch_size, d_A, d_B, d_C_custom);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float custom_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&custom_ms, start, stop));
    const double custom_us = custom_ms * 1000.0 / repeats;

    CUDA_CHECK(cudaMemcpy(h_custom.data(), d_C_custom, bytes, cudaMemcpyDeviceToHost));

    // --------------------------------------------------------
    // cuBLAS strided batched GEMM
    // Data is already column-major, so no operand swap is needed:
    // C = A * B with OP_N, OP_N.
    // --------------------------------------------------------
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const long long stride = (long long)matrix_size;

    // Warmup run for cuBLAS
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle,
                                           CUBLAS_OP_N, CUBLAS_OP_N,
                                           n, n, n,
                                           &alpha,
                                           d_A, n, stride, // A
                                           d_B, n, stride, // B
                                           &beta,
                                           d_C_cublas, n, stride, // C
                                           batch_size));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timing cuBLAS
    CUDA_CHECK(cudaEventRecord(start, 0));
    for (int r = 0; r < repeats; ++r)
    {
        CUBLAS_CHECK(cublasSgemmStridedBatched(handle,
                                               CUBLAS_OP_N, CUBLAS_OP_N,
                                               n, n, n,
                                               &alpha,
                                               d_A, n, stride, // A
                                               d_B, n, stride, // B
                                               &beta,
                                               d_C_cublas, n, stride, // C
                                               batch_size));
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float cublas_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&cublas_ms, start, stop));
    const double cublas_us = cublas_ms * 1000.0 / repeats;

    CUDA_CHECK(cudaMemcpy(h_cublas.data(), d_C_cublas, bytes, cudaMemcpyDeviceToHost));

    // --------------------------------------------------------
    // Performance and forward error
    // --------------------------------------------------------
    const double flops = 2.0 * n * n * n * batch_size;
    const double custom_gflops = flops / (custom_us * 1.0e3);
    const double cublas_gflops = flops / (cublas_us * 1.0e3);

    const double custom_error = max_batch_forward_error(n, batch_size, h_custom.data(), h_ref.data());
    const double cublas_error = max_batch_forward_error(n, batch_size, h_cublas.data(), h_ref.data());

    std::printf("%-24s %12s %12s %20s\n", "Implementation", "Time (us)", "GFLOP/s", "Max batch fwd error");
    std::printf("-----------------------------------------------------------------------\n");
    std::printf("%-24s %12.3f %12.3f %20.6e\n", "Custom kernel", custom_us, custom_gflops, custom_error);
    std::printf("%-24s %12.3f %12.3f %20.6e\n", "cuBLAS Strided Batched", cublas_us, cublas_gflops, cublas_error);

    const double tol = 1e-6;
    if (custom_error <= tol && cublas_error <= tol)
    {
        std::printf("\nTEST PASS\n");
        std::printf("Speedup (cuBLAS time / custom time): %.3fx\n", cublas_us / custom_us);
    }
    else
    {
        std::printf("\nTEST FAIL\n");
        std::printf("Speedup is not calculated\n");
    }

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_custom));
    CUDA_CHECK(cudaFree(d_C_cublas));

    return 0;
}
