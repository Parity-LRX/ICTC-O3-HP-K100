#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

template<bool Unit>
__global__ void cached_geometry_kernel(const float* x,const float* w,const float* g,
    const int64_t* src,const int64_t* dst,const float* mask,const float* v,
    const int32_t* bm,const int32_t* cp,float* output,int c,int xdim,int odim,int bdim,int p,bool masked){
    const int64_t e=blockIdx.x,s=src[e],d=dst[e];const int t=threadIdx.x;
    const int lane=t%32,warp=t/32,warps=blockDim.x/32;
    extern __shared__ float memory[];float* sx=memory;float* sv=sx+xdim;
    for(int i=t;i<xdim;i+=blockDim.x)sx[i]=x[s*xdim+i];
    for(int i=t;i<odim;i+=blockDim.x){
        const int path=cp[i/c],channel=i%c;
        float value=v[d*odim+i];if(masked)value*=mask[e];
        value*=g[(e*p+path)*c+channel];
        if(!Unit)value*=w[path*c+channel];
        sv[i]=value;
    }
    __syncthreads();
    for(int col=warp;col<bdim;col+=warps){
        const int xp=bm[bdim+col],ob=bm[3*bdim+col];float value=0.f;
        for(int channel=lane;channel<c;channel+=32)value+=sx[xp+channel]*sv[ob+channel];
        #pragma unroll
        for(int shift=16;shift>0;shift/=2)value+=__shfl_down_sync(0xffffffff,value,shift);
        if(lane==0)output[e*bdim+col]=value;
    }
}

torch::Tensor cached_edge_geometry_cuda(torch::Tensor x,torch::Tensor w,torch::Tensor g,
    torch::Tensor src,torch::Tensor dst,torch::Tensor mask,torch::Tensor v,
    torch::Tensor bm,torch::Tensor cp,bool unit){
    const c10::cuda::CUDAGuard guard(x.device());
    for(const auto& t:{x,w,g,mask,v})TORCH_CHECK(t.is_cuda()&&t.device()==x.device()&&t.scalar_type()==at::kFloat&&t.is_contiguous(),"Invalid cached geometry FP32 operands");
    for(const auto& t:{src,dst})TORCH_CHECK(t.is_cuda()&&t.device()==x.device()&&t.scalar_type()==at::kLong&&t.is_contiguous(),"Invalid geometry topology");
    for(const auto& t:{bm,cp})TORCH_CHECK(t.is_cuda()&&t.device()==x.device()&&t.scalar_type()==at::kInt&&t.is_contiguous(),"Invalid geometry metadata");
    TORCH_CHECK(bm.dim()==2&&bm.size(0)==5&&x.dim()==2&&v.dim()==2&&w.dim()==2&&g.dim()==3,"Invalid geometry shapes");
    const int c=w.size(1),xdim=x.size(1),odim=v.size(1),bdim=bm.size(1),p=w.size(0);
    TORCH_CHECK(c>0&&c<=256&&odim%c==0&&cp.numel()==odim/c&&g.size(0)==src.numel()&&dst.numel()==src.numel(),"Inconsistent geometry shapes");
    TORCH_CHECK(mask.numel()==0||mask.numel()==src.numel(),"Invalid geometry mask");
    const int bytes=(xdim+odim)*sizeof(float);TORCH_CHECK(bytes<=48000,"Geometry shared storage exceeds supported size");
    auto result=torch::empty({src.numel(),bdim},x.options());if(!src.numel())return result;
    auto stream=at::cuda::getCurrentCUDAStream();
    #define LAUNCH(FLAG) cached_geometry_kernel<FLAG><<<src.numel(),256,bytes,stream>>>(x.data_ptr<float>(),w.data_ptr<float>(),g.data_ptr<float>(),src.data_ptr<int64_t>(),dst.data_ptr<int64_t>(),mask.data_ptr<float>(),v.data_ptr<float>(),bm.data_ptr<int32_t>(),cp.data_ptr<int32_t>(),result.data_ptr<float>(),c,xdim,odim,bdim,p,mask.numel()!=0)
    if(unit){LAUNCH(true);}else{LAUNCH(false);}
    #undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();return result;
}
