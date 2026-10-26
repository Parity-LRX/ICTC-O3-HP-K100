"""HIP implementation of the existing one-output message layout copy."""
import hashlib
from functools import lru_cache
import torch


@lru_cache(maxsize=128)
def _extension(specs,to_packed):
    body=[]
    for offset in range(0,len(specs),4):
        base,count,magnetic,channels=specs[offset:offset+4];span=magnetic*channels
        mapped=(f'(inner%{channels})*{magnetic}+inner/{channels}' if to_packed
                else f'(inner%{magnetic})*{channels}+inner/{magnetic}')
        body.append(f'''{{const int64_t local=column-{base};
 if(local>=0&&local<{count*span}) {{const int64_t path=local/{span},inner=local%{span};
 source=row*width+{base}+path*{span}+{mapped};}}}}''')
    source='''#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
__global__ void copy_kernel(const float* X,float* Y,int64_t elements,int64_t width) {
 const int64_t index=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
 if(index>=elements)return;
 const int64_t row=index/width,column=index%width;int64_t source=0;
'''+ '\n'.join(body)+'''
 Y[index]=X[source];
}
torch::Tensor copy_layout(torch::Tensor x) {
 const c10::cuda::CUDAGuard guard(x.device());
 TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kFloat&&x.is_contiguous()&&x.dim()==2,"Invalid layout tensor");
 auto out=torch::empty_like(x);if(!x.numel())return out;
 auto stream=at::cuda::getCurrentCUDAStream();
 copy_kernel<<<(x.numel()+255)/256,256,0,stream>>>(x.data_ptr<float>(),out.data_ptr<float>(),x.numel(),x.size(1));
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
'''
    from ictc_o3.backends.k100.native_jit import load_inline
    name='ictc_hip_layout_'+hashlib.sha256(source.encode()).hexdigest()[:16]
    return load_inline(name=name,cpp_sources='#include <torch/extension.h>\ntorch::Tensor copy_layout(torch::Tensor);',
                       cuda_sources=source,functions=['copy_layout'],extra_cflags=['-O3'],extra_cuda_cflags=['-O3'])


def copy(x,specs,to_packed):
    return _extension(tuple(specs),to_packed).copy_layout(x)
