# SPDX-License-Identifier: MIT
"""Bounded-memory K100 B-spline mesh assignment and its first two derivatives.

No atom-by-stencil-by-channel tensor is materialized. FFTs remain in the
vendor-backed torch.fft implementation. Coordinates are fractional, so the
ordinary Torch position/cell transform also supplies strain derivatives.
"""
from functools import lru_cache
import torch
from torch.autograd.function import once_differentiable
from mace_ictc.utils.mesh import mesh_shape


def supported(value):
    if (torch.are_deterministic_algorithms_enabled() or not torch.version.hip
            or not value.is_cuda or value.dtype not in (torch.float32, torch.float64)):
        return False
    if torch.compiler.is_compiling() or torch.jit.is_tracing():
        return False
    from torch._subclasses.fake_tensor import FakeTensor
    from torch.fx.experimental.proxy_tensor import get_proxy_mode
    if isinstance(value, FakeTensor) or get_proxy_mode() is not None:
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    return getattr(torch.cuda.get_device_properties(value.device), 'gcnArchName', '').split(':')[0] == 'gfx926'


@lru_cache(maxsize=1)
def _extension():
    from .native_jit import load_inline
    return load_inline(
        name='ictc_hip_electrostatic_mesh_v1',
        cpp_sources=r'''
#include <torch/extension.h>
void run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
         torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,
         bool,bool,bool,bool,bool);
''',
        cuda_sources=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cassert>

template<typename T,int O> __device__ inline void spline(T x,int64_t& base,T* w,T* d,T* h) {
 if constexpr(O==2) {
  base=static_cast<int64_t>(floor(x)); const T t=x-T(base);
  w[0]=T(1)-t;w[1]=t;d[0]=-T(1);d[1]=T(1);h[0]=h[1]=T(0);
 } else if constexpr(O==3) {
  base=static_cast<int64_t>(floor(x-T(.5)));const T t=x-T(base);
  w[0]=T(.5)*(T(1.5)-t)*(T(1.5)-t);w[1]=T(.75)-(t-T(1))*(t-T(1));
  w[2]=T(.5)*(t-T(.5))*(t-T(.5));
  d[0]=t-T(1.5);d[1]=T(2)*(T(1)-t);d[2]=t-T(.5);
  h[0]=T(1);h[1]=-T(2);h[2]=T(1);
 } else {
  const int64_t f=static_cast<int64_t>(floor(x));base=f-1;const T t=x-T(f),u=T(1)-t,t2=t*t,t3=t2*t;
  w[0]=u*u*u/T(6);w[1]=(T(3)*t3-T(6)*t2+T(4))/T(6);
  w[2]=(-T(3)*t3+T(3)*t2+T(3)*t+T(1))/T(6);w[3]=t3/T(6);
  d[0]=-u*u/T(2);d[1]=T(1.5)*t2-T(2)*t;d[2]=-T(1.5)*t2+t+T(.5);d[3]=t2/T(2);
  h[0]=u;h[1]=T(3)*t-T(2);h[2]=-T(3)*t+T(1);h[3]=t;
 }
}
__device__ inline int64_t wrap_index(int64_t i,int64_t m) {i%=m;return i<0?i+m:i;}

// A warp traverses neighboring stencil entries for one atom. All source
// components reuse those indices, and channel-first FFT meshes receive
// spatially adjacent writes instead of one strided plane per lane.
template<typename T,int O> __global__ void spread_stencil(
 const T* frac,const int64_t* batch,const T* source,T* output,
 int64_t n,int64_t c,int64_t mx,int64_t my,int64_t mz,int64_t b,
 int64_t x_start,int64_t x_count,bool slab,bool channel_first) {
 const int64_t t=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
 constexpr int stencil=O*O*O;
 if(t>=n*stencil)return;
 const int64_t atom=t/stencil,graph=batch[atom];const int local=t%stencil;
 CUDA_KERNEL_ASSERT(graph>=0 && graph<b);
 const int u[3]={local/(O*O),(local/O)%O,local%O};
 const int64_t sizes[3]={mx,my,mz};
 int64_t index[3];T weight=T(1);
 #pragma unroll
 for(int a=0;a<3;++a) {
  const int64_t m=sizes[a];
  T w[O],d[O],h[O];int64_t base;
  spline<T,O>(frac[3*atom+a]*T(m),base,w,d,h);
  weight*=w[u[a]];const int64_t value=base+u[a];
  index[a]=(slab&&a==2)?(value<0?0:(value>=m?m-1:value)):wrap_index(value,m);
 }
 if(index[0]<x_start||index[0]>=x_start+x_count)return;
 const int64_t points=x_count*my*mz,spatial=((index[0]-x_start)*my+index[1])*mz+index[2];
 for(int64_t ch=0;ch<c;++ch) {
  const int64_t out=channel_first?((graph*c+ch)*points+spatial):((graph*points+spatial)*c+ch);
  atomicAdd(output+out,source[atom*c+ch]*weight);
 }
}

// Modes: spread, gather, coordinate VJP, coordinate-VJP backward.
template<typename T,int O,int Mode> __global__ void assignment(
 const T* frac,const int64_t* batch,const T* source,const T* mesh,const T* direction,
 T* out_source,T* out_mesh,T* out_frac,int64_t n,int64_t c,int64_t mx,int64_t my,int64_t mz,int64_t b,
 int64_t x_start,int64_t x_count,bool slab,bool channel_first,bool need_frac,bool need_source,bool need_mesh) {
 const int64_t t=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
 if(t>=n*c)return;
 const int64_t atom=t/c,ch=t%c,graph=batch[atom];
 CUDA_KERNEL_ASSERT(graph>=0 && graph<b);
 const int64_t sizes[3]={mx,my,mz};
 T w[3][O],d[3][O],h[3][O];int64_t base[3],neighbor[3][O];
 #pragma unroll
 for(int a=0;a<3;++a) {
  const int64_t m=sizes[a];
  spline<T,O>(frac[3*atom+a]*T(m),base[a],w[a],d[a],h[a]);
  #pragma unroll
  for(int z=0;z<O;++z) {
   d[a][z]*=T(m);h[a][z]*=T(m)*T(m);
   const int64_t value=base[a]+z;
   neighbor[a][z]=(slab&&a==2)?(value<0?0:(value>=m?m-1:value)):wrap_index(value,m);
  }
 }
 const T s=(Mode==0||Mode==2||Mode==3)?source[t]:T(0);
 T gathered=0,grad[3]={0,0,0},dir[3]={0,0,0};
 if constexpr(Mode==3) {for(int a=0;a<3;++a)dir[a]=direction[3*atom+a];}
 for(int i=0;i<O;++i) {
  const int64_t x=neighbor[0][i];
  if(x<x_start||x>=x_start+x_count)continue;
  for(int j=0;j<O;++j)for(int k=0;k<O;++k) {
  const int u[3]={i,j,k};
  const int64_t y=neighbor[1][j],z=neighbor[2][k];
  const int64_t spatial=((x-x_start)*my+y)*mz+z,points=x_count*my*mz;
  const int64_t index=channel_first?((graph*c+ch)*points+spatial):((graph*points+spatial)*c+ch);
  const T weight=w[0][i]*w[1][j]*w[2][k];
  if constexpr(Mode==0) {atomicAdd(out_mesh+index,s*weight);}
  if constexpr(Mode==1) {gathered+=mesh[index]*weight;}
  if constexpr(Mode>=2) {
   T dw[3]={d[0][i]*w[1][j]*w[2][k],w[0][i]*d[1][j]*w[2][k],w[0][i]*w[1][j]*d[2][k]};
   const T field=mesh[index];
   if constexpr(Mode==2) {for(int a=0;a<3;++a)grad[a]+=s*field*dw[a];}
   if constexpr(Mode==3) {
    const T directional=dir[0]*dw[0]+dir[1]*dw[1]+dir[2]*dw[2];
    if(need_source)gathered+=field*directional;
    if(need_mesh)atomicAdd(out_mesh+index,s*directional);
    if(need_frac)for(int a=0;a<3;++a)for(int e=0;e<3;++e) {
     T dd=T(1);
     #pragma unroll
     for(int v=0;v<3;++v)dd*=(v==a&&v==e)?h[v][u[v]]:((v==a||v==e)?d[v][u[v]]:w[v][u[v]]);
     grad[a]+=s*field*dir[e]*dd;
    }
   }
  }
 }}
 if constexpr(Mode==1)out_source[t]=gathered;
 if constexpr(Mode==2) {for(int a=0;a<3;++a)atomicAdd(out_frac+3*atom+a,grad[a]);}
 if constexpr(Mode==3) {
  if(need_source)out_source[t]=gathered;
  if(need_frac)for(int a=0;a<3;++a)atomicAdd(out_frac+3*atom+a,grad[a]);
 }
}
void run(torch::Tensor f,torch::Tensor batch,torch::Tensor s,torch::Tensor mesh,torch::Tensor direction,
 torch::Tensor os,torch::Tensor om,torch::Tensor of,int64_t mx,int64_t my,int64_t mz,int64_t b,int64_t order,int64_t mode,
 int64_t x_start,int64_t x_count,
 bool slab,bool cf,bool nf,bool ns,bool nm) {
 const c10::cuda::CUDAGuard guard(f.device());
 TORCH_CHECK(f.dim()==2&&f.size(1)==3&&s.dim()==2&&s.size(0)==f.size(0),"Expected frac[N,3], source[N,C]");
 TORCH_CHECK(batch.scalar_type()==at::kLong&&batch.numel()==f.size(0),"Expected int64 batch[N]");
 TORCH_CHECK(mx>0&&my>0&&mz>0&&b>0&&s.size(1)>0&&order>=2&&order<=4&&mode>=0&&mode<=3,"Invalid mesh assignment configuration");
 TORCH_CHECK(x_start>=0&&x_count>0&&mesh.numel()==b*x_count*my*mz*s.size(1),"Invalid mesh slab size");
 for(const auto& t:{f,s,mesh,direction,os,om,of})TORCH_CHECK(t.is_cuda()&&t.device()==f.device()&&t.scalar_type()==f.scalar_type()&&t.is_contiguous(),"Mesh tensors must share device/dtype and be contiguous");
 TORCH_CHECK(batch.is_cuda()&&batch.device()==f.device()&&batch.is_contiguous(),"Invalid batch storage");
 const int64_t n=f.size(0),c=s.size(1);if(n==0)return;
 if(mode==0||nm)TORCH_CHECK(om.numel()==mesh.numel(),"Invalid spread output");
 if(mode==1||ns)TORCH_CHECK(os.numel()==n*c,"Invalid gather output");
 if(mode==2||nf)TORCH_CHECK(of.numel()==n*3,"Invalid coordinate output");
 if(mode==3)TORCH_CHECK(direction.numel()==n*3,"Invalid coordinate direction");
 const auto stream=at::cuda::getCurrentCUDAStream();
 #define LAUNCH(O,M) assignment<scalar_t,O,M><<<(n*c+255)/256,256,0,stream>>>(f.data_ptr<scalar_t>(),batch.data_ptr<int64_t>(),s.data_ptr<scalar_t>(),mesh.data_ptr<scalar_t>(),direction.data_ptr<scalar_t>(),os.data_ptr<scalar_t>(),om.data_ptr<scalar_t>(),of.data_ptr<scalar_t>(),n,c,mx,my,mz,b,x_start,x_count,slab,cf,nf,ns,nm)
 #define SPREAD(O) spread_stencil<scalar_t,O><<<(n*(O*O*O)+255)/256,256,0,stream>>>(f.data_ptr<scalar_t>(),batch.data_ptr<int64_t>(),s.data_ptr<scalar_t>(),om.data_ptr<scalar_t>(),n,c,mx,my,mz,b,x_start,x_count,slab,cf)
 #define MODES(O) if(mode==0){if(cf){SPREAD(O);}else{LAUNCH(O,0);}}else if(mode==1){LAUNCH(O,1);}else if(mode==2){LAUNCH(O,2);}else{LAUNCH(O,3);}
 AT_DISPATCH_FLOATING_TYPES(f.scalar_type(),"ictc_mesh_assignment",[&]{
  if(order==2){MODES(2);}else if(order==3){MODES(3);}else{MODES(4);}
 });
 #undef MODES
 #undef SPREAD
 #undef LAUNCH
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
''', functions=['run'], extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '-ffp-contract=off'])


def _shape(config, channels):
    m, b, _, _, _, nx, cf = config
    shape = (nx, m[1], m[2])
    return (b, channels, *shape) if cf else (b, *shape, channels)


def _run(frac, batch, source, mesh, config, mode, direction=None, needed=(True, True, True)):
    m, b, order, slab, x_start, x_count, cf = config
    empty = frac.new_empty(0)
    out_source = torch.empty_like(source) if mode == 1 or (mode == 3 and needed[1]) else empty
    out_mesh = mesh.zero_() if mode == 0 else (torch.zeros_like(mesh) if mode == 3 and needed[2] else empty)
    out_frac = torch.zeros_like(frac) if mode == 2 or (mode == 3 and needed[0]) else empty
    _extension().run(frac, batch, source, mesh, empty if direction is None else direction.contiguous(),
                     out_source, out_mesh, out_frac, *m, b, order, mode, x_start, x_count, slab, cf,
                     *(needed if mode == 3 else (False, False, False)))
    return out_frac, out_source, out_mesh


class _Spread(torch.autograd.Function):
    @staticmethod
    def forward(ctx, frac, batch, source, config):
        ctx.save_for_backward(frac, batch, source)
        ctx.config = config
        mesh = source.new_empty(_shape(config, source.shape[1]))
        return _run(frac, batch, source, mesh, config, 0)[2]

    @staticmethod
    def backward(ctx, grad):
        frac, batch, source = ctx.saved_tensors
        grad = grad.contiguous()
        df = _CoordinateVJP.apply(frac, batch, source, grad, ctx.config) if ctx.needs_input_grad[0] else None
        ds = _Gather.apply(frac, batch, grad, ctx.config) if ctx.needs_input_grad[2] else None
        return df, None, ds, None


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, frac, batch, mesh, config):
        ctx.save_for_backward(frac, batch, mesh)
        ctx.config = config
        channels = mesh.shape[1] if config[-1] else mesh.shape[-1]
        source = frac.new_empty((frac.shape[0], channels))
        return _run(frac, batch, source, mesh, config, 1)[1]

    @staticmethod
    def backward(ctx, grad):
        frac, batch, mesh = ctx.saved_tensors
        grad = grad.contiguous()
        df = _CoordinateVJP.apply(frac, batch, grad, mesh, ctx.config) if ctx.needs_input_grad[0] else None
        dm = _Spread.apply(frac, batch, grad, ctx.config) if ctx.needs_input_grad[2] else None
        return df, None, dm, None


class _CoordinateVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, frac, batch, source, mesh, config):
        ctx.save_for_backward(frac, batch, source, mesh)
        ctx.config = config
        return _run(frac, batch, source, mesh, config, 2)[0]

    @staticmethod
    @once_differentiable
    def backward(ctx, direction):
        frac, batch, source, mesh = ctx.saved_tensors
        df, ds, dm = _run(frac, batch, source, mesh, ctx.config, 3, direction,
                          needed=(ctx.needs_input_grad[0], ctx.needs_input_grad[2], ctx.needs_input_grad[3]))
        return df if ctx.needs_input_grad[0] else None, None, ds if ctx.needs_input_grad[2] else None, dm if ctx.needs_input_grad[3] else None, None


def _config(frac, batch, mesh_size, num_graphs, assignment, boundary, channels_first,
            x_start=0, x_count=None):
    if torch.are_deterministic_algorithms_enabled():
        raise ValueError('Native mesh assignment uses atomic reductions; select the reference backend for deterministic execution')
    if not supported(frac):
        raise ValueError('Native mesh assignment requires K100 gfx926 FP32 or FP64')
    if frac.ndim != 2 or frac.shape[1] != 3 or batch.shape != frac.shape[:1]:
        raise ValueError('Expected fractional coordinates [N,3] and batch [N]')
    if batch.dtype != torch.int64 or batch.device != frac.device:
        raise ValueError('batch must be int64 on the coordinate device')
    if assignment not in ('cic', 'tsc', 'pcs') or boundary not in ('periodic', 'slab'):
        raise ValueError('Expected cic/tsc/pcs assignment and periodic/slab boundary')
    shape = mesh_shape(mesh_size)
    if x_count is None:
        x_count = shape[0]
    if not isinstance(x_start, int) or not isinstance(x_count, int) or x_start < 0 or x_count < 1:
        raise ValueError('Expected a nonnegative x_start and positive x_count')
    if not isinstance(num_graphs, int) or isinstance(num_graphs, bool) or num_graphs < 1:
        raise ValueError('num_graphs must be a positive integer')
    return shape, num_graphs, {'cic': 2, 'tsc': 3, 'pcs': 4}[assignment], boundary == 'slab', x_start, x_count, bool(channels_first)


def spread(frac, batch, source, *, mesh_size, num_graphs, assignment='pcs', boundary='periodic', channels_first=False,
           x_start=0, x_count=None):
    config = _config(frac, batch, mesh_size, num_graphs, assignment, boundary, channels_first, x_start, x_count)
    if source.ndim != 2 or source.shape[0] != frac.shape[0] or source.shape[1] < 1:
        raise ValueError('Expected source [N,C] with C > 0')
    if source.device != frac.device or source.dtype != frac.dtype:
        raise ValueError('Source and coordinates must share device and dtype')
    return _Spread.apply(frac.contiguous(), batch.contiguous(), source.contiguous(), config)


def gather(frac, batch, mesh, *, mesh_size, num_graphs, assignment='pcs', boundary='periodic', channels_first=False,
           x_start=0, x_count=None):
    config = _config(frac, batch, mesh_size, num_graphs, assignment, boundary, channels_first, x_start, x_count)
    if mesh.ndim != 5:
        raise ValueError('Expected a five-dimensional mesh')
    channels = mesh.shape[1] if channels_first and mesh.ndim == 5 else mesh.shape[-1]
    if mesh.shape != _shape(config, channels) or mesh.device != frac.device or mesh.dtype != frac.dtype:
        raise ValueError('Invalid mesh shape, device or dtype')
    return _Gather.apply(frac.contiguous(), batch.contiguous(), mesh.contiguous(), config)
