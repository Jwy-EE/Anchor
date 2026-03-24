#ifndef DCNV4_1D_KERNEL_CUH
#define DCNV4_1D_KERNEL_CUH
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
constexpr int WARP_SIZE = 32;
constexpr int MAX_BLOCK_SIZE = 256;
constexpr int SHARED_MEM_SIZE = 48 * 1024;
__device__ __forceinline__ float bilinear_interp_1d(
    const float* data,
    int length,
    float x
) {
    int x0 = __float2int_rd(x);
    int x1 = x0 + 1;
    float dx = x - x0;
    x0 = max(0, min(length - 1, x0));
    x1 = max(0, min(length - 1, x1));
    float v0 = data[x0];
    float v1 = data[x1];
    return v0 * (1.0f - dx) + v1 * dx;
}
__global__ void dcnv4_sampling_kernel(
    const float* __restrict__ input,
    const float* __restrict__ offset,
    const float* __restrict__ mask,
    float* __restrict__ output,
    const int N,
    const int C,
    const int L,
    const int G,
    const int K,
    const float offset_scale,
    const int dilation
);
__global__ void dcnv4_optimized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ offset,
    const float* __restrict__ mask,
    float* __restrict__ output,
    const int N,
    const int C,
    const int L,
    const int G,
    const int K,
    const float offset_scale,
    const int dilation
);
template<typename T>
__global__ void dcnv4_fused_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ weight_input,
    const T* __restrict__ weight_offset,
    const T* __restrict__ weight_mask,
    const T* __restrict__ weight_output,
    const int N,
    const int C,
    const int L,
    const int G,
    const int K,
    const float offset_scale,
    const int dilation,
    const bool use_half
);
struct KernelConfig {
    dim3 grid_dim;
    dim3 block_dim;
    int shared_mem_size;
};
KernelConfig compute_kernel_config(
    int N, int C, int L, int G, int K,
    int stage = 1
);
cudaError_t launch_dcnv4_kernel(
    const float* input,
    const float* offset,
    const float* mask,
    float* output,
    int N, int C, int L, int G, int K,
    float offset_scale,
    int dilation,
    int stage = 1,
    cudaStream_t stream = 0
);
#endif
