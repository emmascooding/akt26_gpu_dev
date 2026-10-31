
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
// Custom rank-1 update kernel
// A <- A + alpha * x * y^T
// Column-major storage: A(i,j) = A[j*m + i]
// threadIdx.x runs along i (rows) => consecutive threads touch consecutive
// addresses of A and x => coalesced accesses.
__global__ void ger_kernel(int m, int n, float alpha, const float* x, const float* y, float* A)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < m && j < n) {
        A[(size_t)j * m + i] += alpha * x[i] * y[j];
    }
}

////////////////////////////////////////////////////////////////////////////////
// CPU reference implementation, FP32
void ger_cpu(int m, int n, float alpha, const float* x, const float* y, float* A)
{
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            A[(size_t)j * m + i] += alpha * x[i] * y[j];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// relative error norm ||result - reference||_2 / ||reference||_2
double compute_error_norm(size_t size,
                          const float* result,
                          const float* reference)
{
    double error = 0.0;
    double norm  = 0.0;

    for (size_t i = 0; i < size; ++i) {
        const double diff  = (double)result[i] - (double)reference[i];
        const double denom = (double)reference[i];

        error += diff * diff;
        norm  += denom * denom;
    }

    return std::sqrt(error) / std::sqrt(norm);
}

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    const int m = (argc > 1) ? std::atoi(argv[1]) : 4000;
    const int n = (argc > 2) ? std::atoi(argv[2]) : 2000;
    const int repeats = (argc > 3) ? std::atoi(argv[3]) : 100;
    const float alpha = 1.5f;

    if (m <= 0 || n <= 0 || repeats <= 0) {
        std::fprintf(stderr, "Usage: %s [M > 0] [N > 0] [repeats > 0]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const size_t matrix_size = (size_t)m * n;
    const size_t matrix_bytes = matrix_size * sizeof(float);

    // --------------------------------------------------------
    // Initialize vectors and matrix on the CPU
    // --------------------------------------------------------
    std::vector<float> h_x(m), h_y(n);
    std::vector<float> h_A0(matrix_size);
    std::vector<float> h_ref(matrix_size);
    std::vector<float> h_custom(matrix_size);
    std::vector<float> h_cublas(matrix_size);

    for (int i = 0; i < m; ++i) {
        h_x[i] = std::sin(0.1f * (i + 1));
    }

    for (int j = 0; j < n; ++j) {
        h_y[j] = std::cos(0.2f * (j + 1));
    }

    for (size_t i = 0; i < matrix_size; ++i) {
        h_A0[i] = 0.1f * std::sin(0.1f * (float)(i + 1));
    }

    h_ref = h_A0;
    ger_cpu(m, n, alpha, h_x.data(), h_y.data(), h_ref.data());

    // --------------------------------------------------------
    // Allocate GPU memory
    // --------------------------------------------------------
    float *d_x = nullptr, *d_y = nullptr, *d_A = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, (size_t)m * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A, matrix_bytes));

    // copy input vectors from CPU -> GPU
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), (size_t)m * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice));

    std::printf("Matrix: %d x %d, alpha = %.2f, repeats = %d\n",
                m, n, alpha, repeats);
    std::printf("Matrix storage: column-major\n\n");

    // --------------------------------------------------------
    // Kernel configuration
    // --------------------------------------------------------
    // x is the fast (row) dimension of the column-major matrix, so the block is
    // wide in x for coalescing: a warp covers 32 consecutive rows of one column.
    // 64 x 4 = 256 threads per block. Other shapes worth benchmarking:
    // (32,8), (128,2), (256,1). Keep the product a multiple of 32 and <= 1024.
    dim3 block(64, 4, 1);
    dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y, 1);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --------------------------------------------------------
    // Custom kernel: warm-up and timing
    // --------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(d_A, h_A0.data(), matrix_bytes, cudaMemcpyHostToDevice));

    ger_kernel<<<grid, block>>>(m, n, alpha, d_x, d_y, d_A);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Restore the initial matrix before the timed repetitions.
    CUDA_CHECK(cudaMemcpy(d_A, h_A0.data(), matrix_bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < repeats; ++r) {
        ger_kernel<<<grid, block>>>(m, n, alpha, d_x, d_y, d_A);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float custom_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&custom_ms, start, stop));
    const double custom_us = custom_ms * 1000.0 / repeats;

    // Correctness run: restore initial A and update once.
    CUDA_CHECK(cudaMemcpy(d_A, h_A0.data(), matrix_bytes, cudaMemcpyHostToDevice));
    ger_kernel<<<grid, block>>>(m, n, alpha, d_x, d_y, d_A);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_custom.data(), d_A, matrix_bytes, cudaMemcpyDeviceToHost));

    // --------------------------------------------------------
    // cuBLAS SGER: warm-up and timing
    // --------------------------------------------------------
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    CUDA_CHECK(cudaMemcpy(d_A, h_A0.data(), matrix_bytes, cudaMemcpyHostToDevice));

    CUBLAS_CHECK(cublasSger(handle, m, n, &alpha, d_x, 1, d_y, 1, d_A, m));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Restore the initial matrix before the timed repetitions.
    CUDA_CHECK(cudaMemcpy(d_A, h_A0.data(), matrix_bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < repeats; ++r) {
        CUBLAS_CHECK(cublasSger(handle, m, n, &alpha, d_x, 1, d_y, 1, d_A, m));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float cublas_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&cublas_ms, start, stop));
    const double cublas_us = cublas_ms * 1000.0 / repeats;

    // Correctness run: restore initial A and update once.
    CUDA_CHECK(cudaMemcpy(d_A, h_A0.data(), matrix_bytes, cudaMemcpyHostToDevice));
    CUBLAS_CHECK(cublasSger(handle, m, n, &alpha, d_x, 1, d_y, 1, d_A, m));
    CUDA_CHECK(cudaMemcpy(h_cublas.data(), d_A, matrix_bytes, cudaMemcpyDeviceToHost));

    // --------------------------------------------------------
    // Performance and error norm
    // Approximately 2 FLOPs per matrix element.
    // --------------------------------------------------------
    const double flops = 2.0 * m * n;

    const double custom_gflops = flops / (custom_us * 1.0e3);
    const double cublas_gflops = flops / (cublas_us * 1.0e3);

    const double custom_error = compute_error_norm(matrix_size, h_custom.data(), h_ref.data());
    const double cublas_error = compute_error_norm(matrix_size, h_cublas.data(), h_ref.data());

    std::printf("%-20s %12s %12s %20s\n",
                "Implementation", "Time (us)", "GFLOP/s",
                "Error Norm");
    std::printf("--------------------------------------------------------------------\n");

    std::printf("%-20s %12.3f %12.3f %20.6e\n",
                "Custom kernel", custom_us, custom_gflops,
                custom_error);
    std::printf("%-20s %12.3f %12.3f %20.6e\n",
                "cuBLAS SGER", cublas_us, cublas_gflops,
                cublas_error);

    const double tol = 1e-6;
    if (custom_error <= tol) {
        std::printf("\nTEST PASS\n");
        std::printf("Speedup (cuBLAS time / custom time): %.3fx\n", cublas_us / custom_us);
    }
    else {
        std::printf("\nTEST FAIL\n");
        std::printf("Speedup is not calculated\n");
    }

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_A));

    return 0;
}
