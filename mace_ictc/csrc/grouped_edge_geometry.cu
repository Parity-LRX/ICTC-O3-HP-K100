#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

template<bool Unit, int BE>
__global__ void grouped_geometry_kernel(const float* x, const float* w, const float* g,
    const int64_t* src, const int64_t* dst, const float* mask, const float* v,
    const int32_t* groups, const int32_t* columns, float* output, int64_t edges,
    int c, int xdim, int odim, int bdim, int p, int group_count, int column_count, bool masked) {
    const int group = blockIdx.x % group_count;
    const int64_t begin = static_cast<int64_t>(blockIdx.x / group_count) * BE;
    const int32_t* row = groups + group * 5;
    const int xb = row[0], ni = row[1], bb = row[2], kt = row[3], base = row[4];
    const int stride = (ni + kt) * c;
    extern __shared__ float memory[];
    for (int index = threadIdx.x; index < BE * stride; index += blockDim.x) {
        const int edge = index / stride, local = index % stride;
        const int64_t e = begin + edge;
        float value = 0.f;
        if (e < edges) {
            if (local < ni * c) {
                value = x[src[e] * xdim + xb + local];
            } else {
                const int k = (local - ni * c) / c, channel = local % c;
                const int path = columns[base + k], ob = columns[column_count + base + k];
                value = v[dst[e] * odim + ob + channel];
                if (masked) value *= mask[e];
                value *= g[(e * p + path) * c + channel];
                if (!Unit) value *= w[path * c + channel];
            }
        }
        memory[index] = value;
    }
    __syncthreads();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int edge = warp % BE, local_warp = warp / BE, per_edge_warps = blockDim.x / (32 * BE);
    const float* sx = memory + edge * stride;
    const float* sv = sx + ni * c;
    const int64_t e = begin + edge;
    for (int column = local_warp; column < ni * kt; column += per_edge_warps) {
        const int i = column / kt, k = column % kt;
        float value = 0.f;
        for (int channel = lane; channel < c; channel += 32)
            value += sx[i * c + channel] * sv[k * c + channel];
        #pragma unroll
        for (int shift = 16; shift > 0; shift /= 2)
            value += __shfl_down_sync(0xffffffff, value, shift);
        if (lane == 0 && e < edges) output[e * bdim + bb + column] = value;
    }
}

torch::Tensor grouped_edge_geometry_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor g,
    torch::Tensor src, torch::Tensor dst, torch::Tensor mask, torch::Tensor v,
    torch::Tensor groups, torch::Tensor columns, int64_t bdim64, bool unit,
    int64_t tile_edges, int64_t warps) {
    const c10::cuda::CUDAGuard guard(x.device());
    for (const auto& t : {x,w,g,mask,v})
        TORCH_CHECK(t.is_cuda() && t.device()==x.device() && t.scalar_type()==at::kFloat && t.is_contiguous(), "Invalid geometry FP32 operand");
    for (const auto& t : {src,dst})
        TORCH_CHECK(t.is_cuda() && t.device()==x.device() && t.scalar_type()==at::kLong && t.is_contiguous(), "Invalid geometry topology");
    for (const auto& t : {groups,columns})
        TORCH_CHECK(t.is_cuda() && t.device()==x.device() && t.scalar_type()==at::kInt && t.is_contiguous(), "Invalid geometry metadata");
    TORCH_CHECK(x.dim()==2 && v.dim()==2 && w.dim()==2 && g.dim()==3, "Invalid geometry operand ranks");
    TORCH_CHECK(groups.dim()==2 && groups.size(1)==5 && groups.size(0)>0 && columns.dim()==2 && columns.size(0)==2, "Invalid geometry layout");
    TORCH_CHECK((tile_edges==1 || tile_edges==2 || tile_edges==4) && (warps==4 || warps==8) && warps>=tile_edges, "Invalid geometry launch choice");
    const int c=w.size(1), xdim=x.size(1), odim=v.size(1), p=w.size(0), bdim=bdim64;
    TORCH_CHECK(c>0 && c<=256 && x.size(0)==v.size(0) && g.size(1)==p && g.size(2)==c && g.size(0)==src.numel() && dst.numel()==src.numel(), "Inconsistent geometry shapes");
    TORCH_CHECK(mask.numel()==0 || mask.numel()==src.numel(), "Invalid geometry mask");
    // A conservative bound avoids synchronizing layout metadata on every call.
    // Each group has at most 2*lmax+1 input components and odim/c output rows.
    // Actual per-group stride is small. The complete metadata layout is bounded
    // to the L<=3/C<=64 experiment until an explicit validated host bound exists.
    TORCH_CHECK(c==64 && xdim<=1024 && odim<=6336 && bdim<=439, "Unsupported grouped geometry layout");
    const int bytes = tile_edges * (7 + 16) * c * sizeof(float);
    TORCH_CHECK(bytes<=48000, "Geometry shared storage exceeds supported size");
    auto result=torch::empty({src.numel(),bdim}, x.options());
    if (!src.numel()) return result;
    const int64_t blocks=((src.numel()+tile_edges-1)/tile_edges)*groups.size(0);
    TORCH_CHECK(blocks<(1LL<<31), "Grouped geometry grid exceeds supported size");
    const dim3 grid(static_cast<unsigned int>(blocks));
    auto stream=at::cuda::getCurrentCUDAStream();
    #define LAUNCH(U,B) grouped_geometry_kernel<U,B><<<grid,warps*32,bytes,stream>>>(x.data_ptr<float>(),w.data_ptr<float>(),g.data_ptr<float>(),src.data_ptr<int64_t>(),dst.data_ptr<int64_t>(),mask.data_ptr<float>(),v.data_ptr<float>(),groups.data_ptr<int32_t>(),columns.data_ptr<int32_t>(),result.data_ptr<float>(),src.numel(),c,xdim,odim,bdim,p,groups.size(0),columns.size(1),mask.numel()!=0)
    if (unit) {
        if (tile_edges==1) { LAUNCH(true,1); } else if (tile_edges==2) { LAUNCH(true,2); } else { LAUNCH(true,4); }
    } else {
        if (tile_edges==1) { LAUNCH(false,1); } else if (tile_edges==2) { LAUNCH(false,2); } else { LAUNCH(false,4); }
    }
    #undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return result;
}
