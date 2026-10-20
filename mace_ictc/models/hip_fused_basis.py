"""Experimental K100 basis/packing fusion around frozen species circuits."""
from functools import lru_cache
import torch


@lru_cache(maxsize=8)
def _extension(size,sizes):
    # Modes: NCD -> packed basis, packed -> block-flat basis,
    # block-flat cotangent -> packed, packed cotangent -> NCD.
    cases=[];offset=0;qoffset=0
    for width in sizes:
        cases.append(f'''if(local>={offset}*C&&local<{offset+width}*C) {{
 const int64_t v=local-{offset}*C,channel=v/{width},mag=v%{width};float value=0;
 #pragma unroll
 for(int j=0;j<{width};++j)value+=X[(n*{sum(sizes)}+{offset}+j)*C+channel]*Q[{qoffset}+mag*{width}+j];
 Y[index]=value;
 }}''')
        offset+=width;qoffset+=width*width
    reverse=[];offset=0;qoffset=0
    for width in sizes:
        reverse.append(f'''if(mag>={offset}&&mag<{offset+width}) {{float value=0;
 #pragma unroll
 for(int j=0;j<{width};++j)value+=X[n*C*{sum(sizes)}+{offset}*C+channel*{width}+j]*Q[{qoffset}+j*{width}+mag-{offset}];
 Y[index]=value;
 }}''')
        offset+=width;qoffset+=width*width
    source=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
template<int Mode> __global__ void transform(const float* X,const float* Q,float* Y,int64_t N,int64_t C,int64_t width) {
 const int64_t index=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
 if(index>=N*C*width)return;const int64_t n=index/(C*width),local=index%(C*width);
 if constexpr(Mode==0) {
 const int64_t mag=local/C,channel=local%C;float value=0;
 #pragma unroll
 for(int j=0;j<INPUT_SIZE;++j)value+=X[(n*C+channel)*INPUT_SIZE+j]*Q[j*INPUT_SIZE+mag];
 Y[index]=value;
 }
 if constexpr(Mode==1) { OUTPUT_CASES }
 if constexpr(Mode==2) {const int64_t mag=local/C,channel=local%C; REVERSE_CASES }
 if constexpr(Mode==3) {
 const int64_t mag=local%INPUT_SIZE,channel=local/INPUT_SIZE;float value=0;
 #pragma unroll
 for(int j=0;j<INPUT_SIZE;++j)value+=X[(n*INPUT_SIZE+j)*C+channel]*Q[mag*INPUT_SIZE+j];
 Y[index]=value;
 }
}
torch::Tensor run(torch::Tensor x,torch::Tensor q,int64_t channels,int64_t mode) {
 const c10::cuda::CUDAGuard guard(x.device());
 TORCH_CHECK(x.is_cuda()&&q.is_cuda()&&x.device()==q.device()&&x.scalar_type()==at::kFloat&&q.scalar_type()==at::kFloat&&x.is_contiguous()&&q.is_contiguous(),"Invalid basis operands");
 TORCH_CHECK(channels>0&&mode>=0&&mode<4,"Invalid basis transform");
 const int64_t n=x.size(0),width=(mode==0||mode==3)?INPUT_SIZE:OUTPUT_SIZE;
 TORCH_CHECK(x.numel()==n*channels*width&&q.numel()==((mode==0||mode==3)?INPUT_SIZE*INPUT_SIZE:QOUT_SIZE),"Invalid basis shapes");
 auto out=mode==1?torch::empty({n,channels*width},x.options()):mode==3?torch::empty({n,channels,width},x.options()):torch::empty({n,width,channels},x.options());
 if(!x.numel())return out;auto stream=at::cuda::getCurrentCUDAStream();const auto blocks=(x.numel()+255)/256;
 #define RUN(M) transform<M><<<blocks,256,0,stream>>>(x.data_ptr<float>(),q.data_ptr<float>(),out.data_ptr<float>(),n,channels,width)
 if(mode==0){RUN(0);}else if(mode==1){RUN(1);}else if(mode==2){RUN(2);}else{RUN(3);}
 #undef RUN
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
'''.replace('OUTPUT_CASES','\n'.join(cases)).replace('REVERSE_CASES','\n'.join(reverse)).replace('INPUT_SIZE',str(size)).replace('OUTPUT_SIZE',str(sum(sizes))).replace('QOUT_SIZE',str(sum(w*w for w in sizes)))
    from .native_jit import load_inline
    return load_inline(name=f'ictc_hip_fused_basis_{size}_'+ '_'.join(map(str,sizes)),
        cpp_sources='#include <torch/extension.h>\ntorch::Tensor run(torch::Tensor,torch::Tensor,int64_t,int64_t);',
        cuda_sources=source,functions=['run'],extra_cflags=['-O3'],extra_cuda_cflags=['-O3'])


class _Fused(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,bank,index,q,qout,plan):
        from .hip_circuit import raw
        ext=_extension(plan['input_size'],tuple(plan['sizes']));channels=x.shape[1]
        packed=ext.run(x.contiguous(),q,channels,0)
        output=raw(packed,bank,plan,0,idx=index,packed_input=True)
        ctx.save_for_backward(x,packed,bank,index,q,qout);ctx.plan=plan;ctx.ext=ext
        return ext.run(output,qout,channels,1)
    @staticmethod
    def backward(ctx,grad):
        from .hip_circuit import raw
        x,packed,bank,index,q,qout=ctx.saved_tensors;plan=ctx.plan
        if torch.is_grad_enabled():
            from .species_circuit import reference
            local=x.clone();output=reference(local@q,bank,index,plan)
            parts=output.split(plan['sizes'],-1)
            output=torch.cat([(part@matrix.T).reshape(x.shape[0],-1) for part,matrix in zip(parts,plan['qout'])],-1)
            dx=torch.autograd.grad(output,local,grad,create_graph=True)[0]
        else:
            gp=ctx.ext.run(grad.contiguous(),qout,x.shape[1],2)
            dxp=raw(packed,bank,plan,1,gp,idx=index,packed_input=True)
            dx=ctx.ext.run(dxp,q,x.shape[1],3)
        return dx,None,None,None,None,None


def fused(x,bank,index,plan):
    if bank.requires_grad or plan['q'].requires_grad or any(q.requires_grad for q in plan['qout']):
        return None
    qout=torch.cat([q.reshape(-1) for q in plan['qout']])
    torch._assert_async(((index>=0)&(index<bank.shape[0])).all(),'Species index outside the coefficient bank')
    return _Fused.apply(x,bank,index.contiguous(),plan['q'].contiguous(),qout,plan)
