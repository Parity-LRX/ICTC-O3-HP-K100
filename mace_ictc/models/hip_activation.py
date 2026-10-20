"""Allocation-bounded FP32 normalized SiLU for the qualified HIP target."""
from functools import lru_cache
import torch

@lru_cache(maxsize=1)
def _extension():
    from .native_jit import load_inline
    return load_inline(name='ictc_hip_normalized_silu_v1',
        cpp_sources='#include <torch/extension.h>\nvoid run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,double,int64_t,bool,bool);',
        cuda_sources=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
template<int Mode> __global__ void normalized_silu(const float* X,const float* G,const float* H,float* Y,float* Z,int64_t N,float scale,bool needx,bool needg) {
 const int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i>=N)return;
 const float x=X[i],s=1.0f/(1.0f+expf(-x));
 if constexpr(Mode==0) { Y[i]=(x*s)*scale; }
 if constexpr(Mode==1) { Y[i]=(G[i]*scale)*(s*(1.0f+x*(1.0f-s))); }
 if constexpr(Mode==2) {
  const float h=H[i];
  if(needx)Y[i]=(h*G[i])*((scale*(s*(1.0f-s)))*(2.0f+x*(1.0f-2.0f*s)));
  if(needg)Z[i]=h*(scale*(s*(1.0f+x*(1.0f-s))));
 }
}
void run(torch::Tensor x,torch::Tensor g,torch::Tensor h,torch::Tensor y,torch::Tensor z,double scale,int64_t mode,bool needx,bool needg) {
 const c10::cuda::CUDAGuard guard(x.device());
 for(const auto& t:{x,g,h,y,z})TORCH_CHECK(t.is_cuda()&&t.device()==x.device()&&t.scalar_type()==at::kFloat&&t.is_contiguous()&&t.numel()==x.numel(),"Invalid HIP SiLU input");
 TORCH_CHECK(mode>=0&&mode<=2,"Invalid HIP SiLU mode");
 const int64_t n=x.numel();if(!n)return;auto stream=at::cuda::getCurrentCUDAStream();
 #define RUN(M) normalized_silu<M><<<(n+255)/256,256,0,stream>>>(x.data_ptr<float>(),g.data_ptr<float>(),h.data_ptr<float>(),y.data_ptr<float>(),z.data_ptr<float>(),n,static_cast<float>(scale),needx,needg)
 if(mode==0) { RUN(0); } else if(mode==1) { RUN(1); } else { RUN(2); }
 #undef RUN
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
''',functions=['run'],extra_cflags=['-O3'],extra_cuda_cflags=['-O3'])

def enabled(x):
    return bool(torch.version.hip and x.is_cuda and
                getattr(torch.cuda.get_device_properties(x.device),'gcnArchName','').split(':')[0]=='gfx926')

def raw(x,scale,mode=0,g=None,h=None,needed=(True,True)):
    y=torch.empty_like(x) if mode!=2 or needed[0] else None
    z=torch.empty_like(x) if mode==2 and needed[1] else None
    if x.numel():
        _extension().run(x,x if g is None else g.contiguous(),x if h is None else h.contiguous(),
                         x if y is None else y,x if z is None else z,float(scale),mode,*needed)
    if mode==2:return y,z
    return y
