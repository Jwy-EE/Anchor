#include "dcnv4_1d_kernel.cuh"
#include <math.h>
#include <algorithm>

__device__ __forceinline__ float compute_sample_position(
    int l, int k, int K, int dilation,
    float offset_val, float offset_scale, int L
) {
    const float base_offset = (k - (K - 1) / 2.0f) * dilation;
    
    float pos = l + base_offset + offset_val * offset_scale;
    
    pos = fmodf(pos + L, L);
    
    return pos;
}
