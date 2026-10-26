#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

template<bool Unit>
__global__ void cached_message_kernel(const float* x,const float* sh,const float* w,const float* g,
    const int64_t* src,const int64_t* dst,const float* mask,const int32_t* fj,const float* fa,
    const int32_t* km,float* output,int channels,int xdim,int hdim,int bdim,int odim,int paths,int components,int terms,bool masked){
    const int64_t edge=blockIdx.x;const int lane=threadIdx.x;
    const int64_t source=src[edge],destination=dst[edge];
    extern __shared__ float cache[];
    float* sx=cache;float* sg=sx+xdim;float* hs=sg+paths*channels;float* bs=hs+hdim;
    for(int i=lane;i<xdim;i+=blockDim.x)sx[i]=x[source*xdim+i];
    for(int i=lane;i<paths*channels;i+=blockDim.x){
        const float value=g[edge*paths*channels+i];sg[i]=Unit?value:value*w[i];
    }
    for(int i=lane;i<hdim;i+=blockDim.x)hs[i]=sh[edge*hdim+i];
    __syncthreads();
    for(int col=lane;col<bdim;col+=blockDim.x){
        float value=0.f;
        for(int k=0;k<terms;++k)value+=hs[fj[col*terms+k]]*fa[col*terms+k];
        bs[col]=value;
    }
    __syncthreads();
    const float factor=masked?mask[edge]:1.f;
    for(int c=lane;c<channels;c+=blockDim.x){
        for(int k=0;k<components;++k){
            const int p=km[k],ni=km[components+k],xb=km[2*components+k];
            const int bb=km[3*components+k],stride=km[4*components+k];
            const int ob=km[5*components+k];
            float value=0.f;
            for(int i=0;i<ni;++i)value+=sx[xb+i*channels+c]*bs[bb+i*stride];
            atomicAdd(output+destination*odim+ob+c,(value*sg[p*channels+c])*factor);
        }
    }
}

void cached_edge_message_cuda(torch::Tensor x,torch::Tensor sh,torch::Tensor w,torch::Tensor g,
    torch::Tensor src,torch::Tensor dst,torch::Tensor mask,torch::Tensor fj,torch::Tensor fa,
    torch::Tensor km,torch::Tensor output,bool unit){
    const c10::cuda::CUDAGuard guard(x.device());
    for(const auto& t:{x,sh,w,g,mask,fa,output})TORCH_CHECK(t.is_cuda()&&t.device()==x.device()&&t.scalar_type()==at::kFloat&&t.is_contiguous(),"Invalid cached-message FP32 operand");
    for(const auto& t:{src,dst})TORCH_CHECK(t.is_cuda()&&t.device()==x.device()&&t.scalar_type()==at::kLong&&t.is_contiguous(),"Invalid cached-message topology");
    for(const auto& t:{fj,km})TORCH_CHECK(t.is_cuda()&&t.device()==x.device()&&t.scalar_type()==at::kInt&&t.is_contiguous(),"Invalid cached-message metadata");
    TORCH_CHECK(km.dim()==2&&km.size(0)==7&&fj.dim()==2,"Invalid cached-message layout");
    const int c=w.size(1),xdim=x.size(1),hdim=sh.size(1),bdim=fj.size(0),odim=output.size(1),paths=w.size(0),components=km.size(1),terms=fj.size(1);
    TORCH_CHECK(c<=256&&xdim%c==0&&g.size(0)==src.numel()&&sh.size(0)==src.numel()&&dst.numel()==src.numel(),"Invalid cached-message shapes");
    if(!src.numel())return;
    const int bytes=(xdim+paths*c+hdim+bdim)*sizeof(float);
    TORCH_CHECK(bytes<=48000,"Cached-message shared storage exceeds supported limit");
    const int threads=c<=64?64:128;
    auto stream=at::cuda::getCurrentCUDAStream();
    #define LAUNCH(FLAG) cached_message_kernel<FLAG><<<src.numel(),threads,bytes,stream>>>(x.data_ptr<float>(),sh.data_ptr<float>(),w.data_ptr<float>(),g.data_ptr<float>(),src.data_ptr<int64_t>(),dst.data_ptr<int64_t>(),mask.data_ptr<float>(),fj.data_ptr<int32_t>(),fa.data_ptr<float>(),km.data_ptr<int32_t>(),output.data_ptr<float>(),c,xdim,hdim,bdim,odim,paths,components,terms,mask.numel()!=0)
    if(unit){LAUNCH(true);}else{LAUNCH(false);}
    #undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
