#include "dcnv4_1d_kernel.cuh"
#include <math.h>
#include <algorithm>
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
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    const int l = blockIdx.z * blockDim.z + threadIdx.z;
    if (n >= N || c >= C || l >= L) return;
    const int g = c / (C / G);
    const int group_channel = c % (C / G);
    float result = 0.0f;
    for (int k = 0; k < K; ++k) {
        const int offset_idx = ((n * L + l) * G + g) * K + k;
        const float offset_val = offset[offset_idx];
        const float base_offset = (k - (K - 1) * 0.5f) * dilation;
        float sample_pos = l + base_offset + offset_val * offset_scale;
        sample_pos = fmodf(sample_pos + L, L);
        const float sampled = bilinear_interp_1d(
            &input[(n * C + c) * L],
            L,
            sample_pos
        );
        const float mask_val = mask[offset_idx];
        result += sampled * mask_val;
    }
    output[(n * C + c) * L + l] = result;
}
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
) {
    extern __shared__ float shared_mem[];
    const int n = blockIdx.x;
    const int l = blockIdx.y;
    const int g = blockIdx.z;
    const int local_c = threadIdx.x;
    const int local_k = threadIdx.y;
    const int group_channels = C / G;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    if (n >= N || l >= L || g >= G) return;
    if (local_c >= group_channels || local_k >= K) return;
    const int c = g * group_channels + local_c;
    const int base_input_idx = (n * C + c) * L;
    for (int load_idx = lane_id; load_idx < L; load_idx += 32) {
        shared_mem[load_idx] = input[base_input_idx + load_idx];
    }
    __syncthreads();
    const int offset_idx = ((n * L + l) * G + g) * K + local_k;
    const float offset_val = offset[offset_idx];
    const float mask_val = mask[offset_idx];
    const float base_offset = (local_k - (K - 1) * 0.5f) * dilation;
    float sample_pos = l + base_offset + offset_val * offset_scale;
    sample_pos = fmodf(sample_pos + L, L);
    int sample_idx = static_cast<int>(roundf(sample_pos));
    sample_idx = max(0, min(L - 1, sample_idx));
    const float sampled = shared_mem[sample_idx];
    float result = sampled * mask_val;
    for (int offset = 16; offset > 0; offset /= 2) {
        result += __shfl_down_sync(0xffffffff, result, offset);
    }
    if (local_k == 0) {
        output[(n * C + c) * L + l] = result;
    }
}
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
) {
    const int n = blockIdx.x;
    const int l = blockIdx.y;
    const int g = blockIdx.z;
    const int local_c = threadIdx.x;
    const int group_channels = C / G;
    if (n >= N || l >= L || g >= G) return;
    if (local_c >= group_channels) return;
    const int c = g * group_channels + local_c;
    T input_val = input[(n * C + c) * L + l];
    T offset_val = T(0);
    T mask_val = T(0);
    for (int k = 0; k < K; ++k) {
        const int weight_offset_idx = (g * K + k) * C + c;
        const int weight_mask_idx = (g * K + k) * C + c;
        offset_val += weight_offset[weight_offset_idx] * input_val;
        mask_val += weight_mask[weight_mask_idx] * input_val;
    }
    T result = T(0);
    for (int k = 0; k < K; ++k) {
        const float base_offset = (k - (K - 1) * 0.5f) * dilation;
        float sample_pos = l + base_offset + float(offset_val) * offset_scale;
        sample_pos = fmodf(sample_pos + L, L);
        int sample_idx = static_cast<int>(roundf(sample_pos));
        sample_idx = max(0, min(L - 1, sample_idx));
        T sampled = input[(n * C + c) * L + sample_idx];
        result += sampled * mask_val;
    }
    T final_result = result * weight_output[c];
    output[(n * C + c) * L + l] = final_result;
}
template __global__ void dcnv4_fused_kernel<float>(
    const float*, float*, const float*, const float*,
    const float*, const float*, int, int, int, int, int,
    float, int, bool);
template __global__ void dcnv4_fused_kernel<__half>(
    const __half*, __half*, const __half*, const __half*,
    const __half*, const __half*, int, int, int, int, int,
    float, int, bool);
KernelConfig compute_kernel_config(
    int N, int C, int L, int G, int K,
    int stage
) {
    KernelConfig config;
    switch (stage) {
        case 1:
            config.block_dim = dim3(8, 8, 4);
            config.grid_dim = dim3(
                (N + config.block_dim.x - 1) / config.block_dim.x,
                (C + config.block_dim.y - 1) / config.block_dim.y,
                (L + config.block_dim.z - 1) / config.block_dim.z
            );
            config.shared_mem_size = 0;
            break;
        case 2:
            config.block_dim = dim3(32, 4, 1);
            config.grid_dim = dim3(N, L, G);
            config.shared_mem_size = 1024 * sizeof(float);
            break;
        case 3:
            config.block_dim = dim3(256, 1, 1);
            config.grid_dim = dim3(N, L, G);
            config.shared_mem_size = 0;
            break;
        default:
            config.block_dim = dim3(1, 1, 1);
            config.grid_dim = dim3(1, 1, 1);
            config.shared_mem_size = 0;
    }
    return config;
}
cudaError_t launch_dcnv4_kernel(
    const float* input,
    const float* offset,
    const float* mask,
    float* output,
    int N, int C, int L, int G, int K,
    float offset_scale,
    int dilation,
    int stage,
    cudaStream_t stream
) {
    KernelConfig config = compute_kernel_config(N, C, L, G, K, stage);
    cudaError_t err = cudaSuccess;
    switch (stage) {
        case 1:
            dcnv4_sampling_kernel<<<config.grid_dim, config.block_dim, 0, stream>>>(
                input, offset, mask, output,
                N, C, L, G, K, offset_scale, dilation
            );
            break;
        case 2:
            dcnv4_optimized_kernel<<<config.grid_dim, config.block_dim, config.shared_mem_size, stream>>>(
                input, offset, mask, output,
                N, C, L, G, K, offset_scale, dilation
            );
            break;
        case 3:
            dcnv4_sampling_kernel<<<config.grid_dim, config.block_dim, 0, stream>>>(
                input, offset, mask, output,
                N, C, L, G, K, offset_scale, dilation
            );
            break;
        default:
            return cudaErrorInvalidValue;
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }
    return cudaStreamSynchronize(stream);
}
