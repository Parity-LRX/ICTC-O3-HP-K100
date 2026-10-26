"""HIP evaluation of the existing Cartesian polynomial and analytic adjoints."""
import hashlib
import torch
from ictc_o3.ops.generated_direction_harmonics import _differentiate
from ictc_o3.ops.generated_direction_harmonics import _monomial

def expression(terms):
    return '+'.join(f'({float(value)!r}f*{_monomial(powers)})' for powers,value in terms) or '0.0f'

def _source(p):
    hd=p['hdim']
    lines=['#include <torch/extension.h>','#include <ATen/cuda/CUDAContext.h>',
           '#include <c10/cuda/CUDAGuard.h>','#include <c10/cuda/CUDAException.h>',
           'template<int MODE> __global__ void harmonics(const float* N,const float* G,const float* H,float* Y,float* DN,float* DG,int64_t E,bool needn,bool needg) {',
           'const int64_t e=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(e>=E)return;']
    for axis,index in zip('xyz',range(3)):
        lines.extend([f'const float {axis}0=1.0f;',f'const float {axis}1=N[e*3+{index}];'])
        for power in range(2,p['lmax']+1):lines.append(f'const float {axis}{power}={axis}{power-1}*{axis}1;')
    lines.append('float d0=0.0f,d1=0.0f,d2=0.0f;')
    for h,terms in enumerate(p['terms']):
        lines.extend(['{',f'if constexpr(MODE==0) {{ Y[e*{hd}+{h}]={expression(terms)}; }} else {{',
                      f'const float g=G[e*{hd}+{h}];'])
        first=[_differentiate(terms,axis) for axis in range(3)]
        for axis in range(3):lines.append(f'const float j{axis}={expression(first[axis])};')
        lines.append('if constexpr(MODE==1) {')
        for axis in range(3):lines.append(f'd{axis}+=g*j{axis};')
        lines.extend(['} else {','const float h0=H[e*3],h1=H[e*3+1],h2=H[e*3+2];',
                      f'if(needg)DG[e*{hd}+{h}]=h0*j0+h1*j1+h2*j2;','if(needn) {'])
        for axis in range(3):
            rhs='+'.join(f'h{other}*({expression(_differentiate(first[axis],other))})' for other in range(3))
            lines.append(f'd{axis}+=g*({rhs});')
        lines.extend(['}','}','}','}'])
    lines.append('if constexpr(MODE!=0) { if(needn) { DN[e*3]=d0;DN[e*3+1]=d1;DN[e*3+2]=d2; } } }')
    lines.append('''
void run(torch::Tensor n,torch::Tensor g,torch::Tensor h,torch::Tensor y,torch::Tensor dn,torch::Tensor dg,int64_t mode,bool needn,bool needg) {
 const c10::cuda::CUDAGuard guard(n.device());
 for(const auto& t:{n,g,h,y,dn,dg})TORCH_CHECK(t.is_cuda()&&t.device()==n.device()&&t.scalar_type()==at::kFloat&&t.is_contiguous(),"Invalid HIP harmonics tensor");
 TORCH_CHECK(n.dim()==2&&n.size(1)==3&&mode>=0&&mode<=2,"Invalid HIP harmonics shape/mode");
 const int64_t e=n.size(0);if(!e)return;auto stream=at::cuda::getCurrentCUDAStream();
 #define RUN(M) harmonics<M><<<(e+255)/256,256,0,stream>>>(n.data_ptr<float>(),g.data_ptr<float>(),h.data_ptr<float>(),y.data_ptr<float>(),dn.data_ptr<float>(),dg.data_ptr<float>(),e,needn,needg)
 if(mode==0) { RUN(0); } else if(mode==1) { RUN(1); } else { RUN(2); }
 #undef RUN
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
''')
    return '\n'.join(lines)

def raw(n,p,mode=0,g=None,h=None,needed=(True,True)):
    if 'hip_extension' not in p:
        from ictc_o3.backends.k100.native_jit import load_inline
        source=_source(p);key=hashlib.sha256(source.encode()).hexdigest()[:16]
        p['hip_extension']=load_inline(name='ictc_hip_harmonics_'+key,
            cpp_sources='#include <torch/extension.h>\nvoid run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,bool,bool);',
            cuda_sources=source,functions=['run'],extra_cflags=['-O3'],extra_cuda_cflags=['-O3'])
    n=n.contiguous();e=n.shape[0]
    y=n.new_empty(e,p['hdim']) if mode==0 else n
    dn=torch.empty_like(n) if mode!=0 else n
    dg=n.new_empty(e,p['hdim']) if mode==2 and needed[1] else n
    p['hip_extension'].run(n,n if g is None else g.contiguous(),n if h is None else h.contiguous(),
                           y,dn,dg,mode,mode==1 or needed[0],needed[1])
    if mode==0:return y
    if mode==1:return dn
    return dn if needed[0] else None,dg if needed[1] else None
