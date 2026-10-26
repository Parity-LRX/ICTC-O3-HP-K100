#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

// Each block owns one destination and one path/magnetic output. Four warps
// split its live edge list and reduce before a unique, non-atomic output write.
// Channels are warp lanes; geometry and topology loads are warp broadcasts.
template <bool Second>
__global__ void csr_message_kernel(
    const float* x, const float* b, const float* w, const float* g,
    const float* hx, const float* hb, const float* hw, const float* hg,
    const int64_t* src, const float* mask, const int64_t* ptr,
    const int64_t* edges, const int32_t* layout, float* output,
    int64_t nodes, int channels, int xdim, int bdim, int odim,
    int paths, int components, unsigned flags, bool masked) {
    const int64_t node = blockIdx.x / components;
    const int k = blockIdx.x % components;
    if (node >= nodes) return;
    const int path = layout[k];
    const int ni = layout[components+k];
    const int xb = layout[2*components+k];
    const int bb = layout[3*components+k];
    const int bs = layout[4*components+k];
    const int ob = layout[5*components+k];
    const int m = layout[6*components+k];
    const int lane = threadIdx.x % 32;
    const int shard = threadIdx.x / 32;
    __shared__ float partial[128];
    const int64_t start = ptr[node], end = ptr[node+1];
    for (int tile = 0; tile < channels; tile += 32) {
        const int channel = tile+lane;
        float sum = 0.f;
        if (channel < channels) {
            const float weight = w[path*channels+channel];
            const float hweight = Second && (flags&4) ? hw[path*channels+channel] : 0.f;
            for (int64_t position = start+shard; position < end; position += 4) {
                const int64_t edge = edges[position];
                const int64_t source = src[edge];
                float y = 0.f, hy = 0.f;
                for (int i = 0; i < ni; ++i) {
                    const int64_t xi = source*xdim+xb+channel*ni+i;
                    const int64_t bi = edge*bdim+bb+i*bs;
                    const float xv = x[xi], bv = b[bi];
                    y += xv*bv;
                    if (Second) {
                        if (flags&1) hy += hx[xi]*bv;
                        if (flags&2) hy += xv*hb[bi];
                    }
                }
                const int64_t gi = (edge*paths+path)*channels+channel;
                const float gate = g[gi];
                float value;
                if (Second) {
                    float factor = hweight*gate;
                    if (flags&8) factor += weight*hg[gi];
                    value = factor*y+(weight*gate)*hy;
                } else {
                    value = (y*weight)*gate;
                }
                if (masked) value *= mask[edge];
                sum += value;
            }
        }
        partial[shard*32+lane] = sum;
        __syncthreads();
        if (shard == 0 && channel < channels) {
            float value = partial[lane]+partial[32+lane];
            value += partial[64+lane];
            value += partial[96+lane];
            output[node*odim+ob+channel*m] = value;
        }
        __syncthreads();
    }
}

torch::Tensor csr_message_cuda(
    torch::Tensor x, torch::Tensor b, torch::Tensor w, torch::Tensor g,
    torch::Tensor hx, torch::Tensor hb, torch::Tensor hw, torch::Tensor hg,
    torch::Tensor src, torch::Tensor mask, torch::Tensor ptr,
    torch::Tensor edges, torch::Tensor layout, int64_t channels,
    int64_t output_dim, int64_t paths, bool second, int64_t flags) {
    const c10::cuda::CUDAGuard device_guard(x.device());
    for (const auto& value : {x,b,w,g,hx,hb,hw,hg,mask}) {
        TORCH_CHECK(value.is_cuda() && value.device() == x.device()
                    && value.scalar_type() == at::kFloat && value.is_contiguous(),
                    "CSR floating inputs must be contiguous FP32 on one CUDA device");
    }
    for (const auto& value : {src,ptr,edges}) {
        TORCH_CHECK(value.is_cuda() && value.device() == x.device()
                    && value.scalar_type() == at::kLong && value.is_contiguous(),
                    "CSR topology must be contiguous int64 on the input CUDA device");
    }
    TORCH_CHECK(layout.is_cuda() && layout.device() == x.device()
                && layout.scalar_type() == at::kInt && layout.is_contiguous()
                && layout.dim() == 2 && layout.size(0) == 7, "Invalid CSR layout");
    TORCH_CHECK(x.dim() == 2 && b.dim() == 2 && w.dim() == 2 && g.dim() == 3,
                "Invalid CSR operand ranks");
    TORCH_CHECK(channels > 0 && channels <= 256 && output_dim > 0,
                "Invalid CSR channel/output dimensions");
    TORCH_CHECK(ptr.numel() == x.size(0)+1 && edges.numel() == src.numel()
                && b.size(0) == src.numel() && g.size(0) == src.numel()
                && g.size(1) == paths && g.size(2) == channels,
                "Inconsistent CSR topology/operand shapes");
    TORCH_CHECK(mask.numel() == 0 || mask.numel() == src.numel(), "Invalid CSR mask");
    auto output = torch::empty({x.size(0),output_dim}, x.options());
    const int64_t count = x.size(0)*layout.size(1);
    if (count) {
        auto stream = at::cuda::getCurrentCUDAStream();
        if (second) {
            csr_message_kernel<true><<<count,128,0,stream>>>(
                x.data_ptr<float>(),b.data_ptr<float>(),w.data_ptr<float>(),g.data_ptr<float>(),
                hx.data_ptr<float>(),hb.data_ptr<float>(),hw.data_ptr<float>(),hg.data_ptr<float>(),
                src.data_ptr<int64_t>(),mask.data_ptr<float>(),ptr.data_ptr<int64_t>(),
                edges.data_ptr<int64_t>(),layout.data_ptr<int32_t>(),output.data_ptr<float>(),
                x.size(0),channels,x.size(1),b.size(1),output_dim,paths,layout.size(1),flags,mask.numel()!=0);
        } else {
            csr_message_kernel<false><<<count,128,0,stream>>>(
                x.data_ptr<float>(),b.data_ptr<float>(),w.data_ptr<float>(),g.data_ptr<float>(),
                hx.data_ptr<float>(),hb.data_ptr<float>(),hw.data_ptr<float>(),hg.data_ptr<float>(),
                src.data_ptr<int64_t>(),mask.data_ptr<float>(),ptr.data_ptr<int64_t>(),
                edges.data_ptr<int64_t>(),layout.data_ptr<int32_t>(),output.data_ptr<float>(),
                x.size(0),channels,x.size(1),b.size(1),output_dim,paths,layout.size(1),flags,mask.numel()!=0);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

template <bool Second>
__global__ void csr_dx_kernel(
    const float* b, const float* w, const float* g, const float* v,
    const float* hb, const float* hw, const float* hg,
    const int64_t* dst, const float* mask, const int64_t* ptr,
    const int64_t* edges, const int32_t* layout, float* output,
    int64_t nodes, int channels, int ni, int bdim, int odim,
    int paths, int components, unsigned flags, bool masked) {
    const int64_t node=blockIdx.x/ni;
    const int i=blockIdx.x%ni;
    if (node>=nodes) return;
    const int lane=threadIdx.x%32, shard=threadIdx.x/32;
    const int64_t start=ptr[node], end=ptr[node+1];
    __shared__ float partial[128];
    for (int tile=0;tile<channels;tile+=32) {
        const int channel=tile+lane;
        float sum=0.f;
        if (channel<channels) {
            for (int64_t position=start+shard;position<end;position+=4) {
                const int64_t edge=edges[position], destination=dst[edge];
                const float edge_mask=masked?mask[edge]:1.f;
                float value=0.f;
                for (int k=0;k<components;++k) {
                    const int p=layout[k], bb=layout[components+k];
                    const int bs=layout[2*components+k], ob=layout[3*components+k];
                    const int64_t bi=edge*bdim+bb+i*bs;
                    const int64_t gi=(edge*paths+p)*channels+channel;
                    const float weight=w[p*channels+channel], gate=g[gi];
                    const float cotangent=v[destination*odim+ob+channel]*edge_mask;
                    const float geometry=b[bi];
                    if (Second) {
                        float factor=0.f;
                        if (flags&4) factor+=hw[p*channels+channel]*gate;
                        if (flags&8) factor+=weight*hg[gi];
                        float term=factor*geometry;
                        if (flags&2) term+=(weight*gate)*hb[bi];
                        value+=cotangent*term;
                    } else {
                        value+=((cotangent*gate)*weight)*geometry;
                    }
                }
                sum+=value;
            }
        }
        partial[shard*32+lane]=sum;
        __syncthreads();
        if (shard==0 && channel<channels) {
            float value=partial[lane]+partial[32+lane];
            value+=partial[64+lane];value+=partial[96+lane];
            output[(node*channels+channel)*ni+i]=value;
        }
        __syncthreads();
    }
}

torch::Tensor csr_dx_cuda(
    torch::Tensor b, torch::Tensor w, torch::Tensor g, torch::Tensor v,
    torch::Tensor hb, torch::Tensor hw, torch::Tensor hg,
    torch::Tensor dst, torch::Tensor mask, torch::Tensor ptr,
    torch::Tensor edges, torch::Tensor layout, int64_t channels,
    int64_t input_components, bool second, int64_t flags) {
    const c10::cuda::CUDAGuard device_guard(b.device());
    for (const auto& value:{b,w,g,v,hb,hw,hg,mask}) {
        TORCH_CHECK(value.is_cuda() && value.device()==b.device() && value.scalar_type()==at::kFloat
                    && value.is_contiguous(),"CSR dx floating inputs must be contiguous FP32 on one device");
    }
    for (const auto& value:{dst,ptr,edges}) {
        TORCH_CHECK(value.is_cuda() && value.device()==b.device() && value.scalar_type()==at::kLong
                    && value.is_contiguous(),"CSR dx topology must be contiguous int64 on one device");
    }
    TORCH_CHECK(layout.is_cuda() && layout.device()==b.device() && layout.scalar_type()==at::kInt
                && layout.is_contiguous() && layout.dim()==2 && layout.size(0)==5,"Invalid CSR dx layout");
    TORCH_CHECK(ptr.numel()==v.size(0)+1 && b.size(0)==dst.numel() && edges.numel()==dst.numel()
                && channels>0 && channels<=256 && input_components>0 && g.dim()==3
                && g.size(0)==dst.numel() && g.size(2)==channels,"Invalid CSR dx dimensions");
    auto output=torch::empty({v.size(0),channels,input_components},b.options());
    const int64_t count=v.size(0)*input_components;
    if (count) {
        auto stream=at::cuda::getCurrentCUDAStream();
        if (second) {
            csr_dx_kernel<true><<<count,128,0,stream>>>(b.data_ptr<float>(),w.data_ptr<float>(),g.data_ptr<float>(),
                v.data_ptr<float>(),hb.data_ptr<float>(),hw.data_ptr<float>(),hg.data_ptr<float>(),dst.data_ptr<int64_t>(),
                mask.data_ptr<float>(),ptr.data_ptr<int64_t>(),edges.data_ptr<int64_t>(),layout.data_ptr<int32_t>(),
                output.data_ptr<float>(),v.size(0),channels,input_components,b.size(1),v.size(1),g.size(1),layout.size(1),flags,mask.numel()!=0);
        } else {
            csr_dx_kernel<false><<<count,128,0,stream>>>(b.data_ptr<float>(),w.data_ptr<float>(),g.data_ptr<float>(),
                v.data_ptr<float>(),hb.data_ptr<float>(),hw.data_ptr<float>(),hg.data_ptr<float>(),dst.data_ptr<int64_t>(),
                mask.data_ptr<float>(),ptr.data_ptr<int64_t>(),edges.data_ptr<int64_t>(),layout.data_ptr<int32_t>(),
                output.data_ptr<float>(),v.size(0),channels,input_components,b.size(1),v.size(1),g.size(1),layout.size(1),flags,mask.numel()!=0);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}
