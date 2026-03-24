#include <torch/extension.h>
#include <vector>
#include <iostream>
#include "dcnv4_1d_kernel.cuh"

torch::Tensor dcnv4_forward(
    torch::Tensor input,
    torch::Tensor offset,
    torch::Tensor mask,
    int kernel_size,
    int group,
    float offset_scale,
    int dilation
) {
    TORCH_CHECK(input.dim() == 3, "input must be a 3D tensor [N, C, L]");
    TORCH_CHECK(offset.dim() == 3, "offset must be a 3D tensor [N, L, G*K]");
    TORCH_CHECK(mask.dim() == 3, "mask must be a 3D tensor [N, L, G*K]");
    
    TORCH_CHECK(input.is_cuda(), "input must be on GPU");
    TORCH_CHECK(offset.is_cuda(), "offset must be on GPU");
    TORCH_CHECK(mask.is_cuda(), "mask must be on GPU");
    
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(offset.scalar_type() == torch::kFloat32, "offset must be float32");
    TORCH_CHECK(mask.scalar_type() == torch::kFloat32, "mask must be float32");
    
    const int N = input.size(0);
    const int C = input.size(1);
    const int L = input.size(2);
    const int G = group;
    const int K = kernel_size;
    
    TORCH_CHECK(offset.size(0) == N, "offset batch dimension mismatch");
    TORCH_CHECK(offset.size(1) == L, "offset length dimension mismatch");
    TORCH_CHECK(offset.size(2) == G * K, "offset channel dimension mismatch");
    
    TORCH_CHECK(mask.size(0) == N, "mask batch dimension mismatch");
    TORCH_CHECK(mask.size(1) == L, "mask length dimension mismatch");
    TORCH_CHECK(mask.size(2) == G * K, "mask channel dimension mismatch");
    
    auto output = torch::zeros_like(input);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    cudaError_t err = launch_dcnv4_kernel(
        input.data_ptr<float>(),
        offset.data_ptr<float>(),
        mask.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, L, G, K,
        offset_scale,
        dilation,
        1,
        stream
    );
    
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA kernel execution failed: ") + cudaGetErrorString(err)
        );
    }
    
    return output;
}

std::vector<torch::Tensor> dcnv4_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor offset,
    torch::Tensor mask,
    int kernel_size,
    int group,
    float offset_scale,
    int dilation
) {
    auto grad_input = torch::zeros_like(input);
    auto grad_offset = torch::zeros_like(offset);
    auto grad_mask = torch::zeros_like(mask);
    
    return {grad_input, grad_offset, grad_mask};
}

void dcnv4_benchmark(
    torch::Tensor input,
    torch::Tensor offset,
    torch::Tensor mask,
    int kernel_size,
    int group,
    float offset_scale,
    int dilation,
    int iterations = 100
) {
    for (int i = 0; i < 10; ++i) {
        auto output = dcnv4_forward(
            input, offset, mask,
            kernel_size, group, offset_scale, dilation
        );
    }
    
    cudaDeviceSynchronize();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        auto output = dcnv4_forward(
            input, offset, mask,
            kernel_size, group, offset_scale, dilation
        );
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start
    ).count();
    
    double avg_time_ms = duration / 1000.0 / iterations;
    
    std::cout << "DCNv4 Benchmark Results:" << std::endl;
    std::cout << "  Iterations: " << iterations << std::endl;
    std::cout << "  Total Time: " << duration / 1000.0 << " ms" << std::endl;
    std::cout << "  Average Time: " << avg_time_ms << " ms/iter" << std::endl;
    
    const int N = input.size(0);
    const int C = input.size(1);
    const int L = input.size(2);
    const int G = group;
    const int K = kernel_size;
    
    long long flops = N * C * L * K * 10;
    double gflops = flops / (avg_time_ms / 1000.0) / 1e9;
    
    std::cout << "  Estimated Performance: " << gflops << " GFLOPS" << std::endl;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &dcnv4_forward, "DCNv4 1D forward");
    m.def("backward", &dcnv4_backward, "DCNv4 1D backward");
    m.def("benchmark", &dcnv4_benchmark, "DCNv4 benchmark");
}
