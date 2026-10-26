"""Live species coefficient banks for correlation-two symmetric contraction."""
import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch._subclasses.fake_tensor import is_fake
try:
    import triton
    import triton.language as tl
except ImportError:
    triton=None


if triton is not None:
    @triton.jit
    def _forward(X,A,B,Z,Y,N,C:tl.constexpr,I:tl.constexpr,M:tl.constexpr,
                 X0:tl.constexpr,X1:tl.constexpr,X2:tl.constexpr,BLOCK:tl.constexpr):
        index=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
        valid=index<N*C*M;n=index//(C*M);c=(index//M)%C;m=index%M
        z=tl.load(Z+n,valid,0);base=(z*C+c)*M+m
        value=tl.full((BLOCK,),0,tl.float32)
        for i in range(I):
            linear=tl.full((BLOCK,),0,tl.float32)
            for j in range(I):
                a=tl.load(A+base*I*I+i*I+j,valid,0.)
                x=tl.load(X+n*X0+c*X1+j*X2,valid,0.)
                linear=linear+a*x
            linear=linear+tl.load(B+base*I+i,valid,0.)
            xi=tl.load(X+n*X0+c*X1+i*X2,valid,0.)
            value=value+linear*xi
        tl.store(Y+index,value,valid)

    @triton.jit
    def _dx(X,A,B,Z,V,DX,N,C:tl.constexpr,I:tl.constexpr,M:tl.constexpr,
            X0:tl.constexpr,X1:tl.constexpr,X2:tl.constexpr,BLOCK:tl.constexpr):
        index=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
        valid=index<N*C*I;n=index//(C*I);c=(index//I)%C;i=index%I
        z=tl.load(Z+n,valid,0);value=tl.full((BLOCK,),0,tl.float32)
        for m in range(M):
            base=(z*C+c)*M+m
            left=tl.full((BLOCK,),0,tl.float32);right=tl.full((BLOCK,),0,tl.float32)
            for j in range(I):
                x=tl.load(X+n*X0+c*X1+j*X2,valid,0.)
                a=tl.load(A+base*I*I+i*I+j,valid,0.)
                at=tl.load(A+base*I*I+j*I+i,valid,0.)
                left=left+a*x;right=right+x*at
            left=left+tl.load(B+base*I+i,valid,0.)
            v=tl.load(V+n*C*M+c*M+m,valid,0.)
            value=value+(v*left+v*right)
        tl.store(DX+index,value,valid)


def _native(x,a,b,idx):
    coefficient=a.index_select(0,idx)
    linear=b.index_select(0,idx)
    first=torch.einsum("ncmij,ncj->ncmi",coefficient,x)+linear
    return torch.einsum("ncmi,nci->ncm",first,x)


class _Polynomial(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,a,b,idx):
        ctx.save_for_backward(x,a,b,idx)
        n,c,i=x.shape;m=a.shape[2]
        out=x.new_empty(n,c,m)
        if out.numel():_forward[(triton.cdiv(out.numel(),128),)](x,a,b,idx,out,n,c,i,m,*x.stride(),128,num_warps=4,enable_fp_fusion=False)
        return out

    @staticmethod
    def backward(ctx,grad):
        x,a,b,idx=ctx.saved_tensors;needed=ctx.needs_input_grad[:3]
        higher=torch.is_grad_enabled()
        if higher or needed[1] or needed[2]:
            with torch.enable_grad():
                out=_native(x,a,b,idx)
                inputs=[value for value,need in zip((x,a,b),needed) if need]
                derivatives=iter(torch.autograd.grad(out,inputs,grad,create_graph=higher,allow_unused=True))
                values=[next(derivatives) if need else None for need in needed]
            return *values,None
        dx=torch.empty_like(x,memory_format=torch.contiguous_format) if needed[0] else None
        if needed[0] and dx.numel():
            grad=grad.contiguous();n,c,i=x.shape;m=a.shape[2]
            _dx[(triton.cdiv(dx.numel(),128),)](x,a,b,idx,grad,dx,n,c,i,m,*x.stride(),128,num_warps=4,enable_fp_fusion=False)
        return dx,None,None,None


def contract_species_bank(module,x,idx):
    """Return None for unsupported cases; coefficient values are never cached."""
    if (triton is None or not x.is_cuda or x.dtype!=torch.float32 or module.correlation!=2
            or x.shape[-1]>16 or x.shape[1]>256 or torch.compiler.is_compiling()
            or torch.is_autocast_enabled(x.device.type) or get_proxy_mode() is not None
            or torch._C._are_functorch_transforms_active() or is_fake(x)):
        return None
    u2=module.U_tensors(2).to(x);u1=module.U_tensors(1).to(x)
    if module.output_lmax==0:u2=u2.unsqueeze(0);u1=u1.unsqueeze(0)
    a=torch.einsum("mijp,zpc->zcmij",u2,module.weights_max.to(x)).contiguous()
    b=torch.einsum("mip,zpc->zcmi",u1,module.weights[0].to(x)).contiguous()
    return _Polynomial.apply(x,a,b,idx.to(device=x.device,dtype=torch.long).contiguous()).reshape(x.shape[0],x.shape[1]*(2*module.output_lmax+1))
