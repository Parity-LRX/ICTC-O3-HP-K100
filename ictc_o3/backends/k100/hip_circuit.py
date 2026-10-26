"""Native HIP CG circuits and mixed training adjoints for K100.

The existing immutable factorization supplies every coefficient. Features and
species weights remain live operands; there are no edge-sized intermediates.
"""
import hashlib
import torch

_EXTENSIONS={}


def _literal(value):
    text=repr(float(value))
    return text+'f'


def _sum(values):
    values=list(values)
    return '+'.join(f'({v})' for v in values) if values else '0.0f'


def _basis_sum(coefficients,values):
    # Eliminate only exact zeros. Small nonzero basis entries are retained.
    return _sum(value if coefficient==1. else f'{_literal(coefficient)}*{value}'
                for coefficient,value in zip(coefficients,values) if coefficient!=0.)


def _source(plan,banked,basis_fused=False):
    pair,final=plan['pair'],plan['final']
    size=plan['input_size'];width=pair['sizes'][3] if pair is not None else 0
    paths=final['sizes'][2];outsize=final['sizes'][3]
    prefixes={};groups={}
    if pair is not None:
        for a,b,_,out,value in pair['entries']:prefixes.setdefault(out,[]).append((a,b,value))
    for left,right,path,out,value in final['entries']:
        groups.setdefault(left,[]).append((right,path,out,value))
    lines=['#include <torch/extension.h>','#include <ATen/cuda/CUDAContext.h>',
           '#include <c10/cuda/CUDAGuard.h>','#include <c10/cuda/CUDAException.h>',
           'template<bool BACK> __global__ void circuit(const float* X,const float* A,const float* G,const int64_t* IDX,float* Y,int64_t N,int64_t C) {',
           'const int64_t row=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;',
           'if(row>=N*C)return;const int64_t n=row/C,c=row%C;',
           f'const int64_t base=({("IDX[n]*C+c" if banked else "c")})*{paths};']
    if basis_fused:
        for i in range(size):lines.append(f'const float u{i}=X[(n*C+c)*{size}+{i}];')
        q=plan['basis_input']
        for i in range(size):
            expr=_basis_sum((row[i] for row in q),(f'u{j}' for j in range(size)))
            lines.append(f'const float x{i}={expr};')
    else:
        for i in range(size):lines.append(f'const float x{i}=X[(n*{size}+{i})*C+c];')
    lines.append(f'const float x{size}=1.0f;')
    for path in range(paths):lines.append(f'const float w{path}=A[base+{path}];')
    lines.append('if constexpr(!BACK) {')
    for out in range(outsize):lines.append(f'float y{out}=0.0f;')
    for left,entries in groups.items():
        expr=f'x{left-width}' if left>=width else _sum(f'{_literal(v)}*x{a}*x{b}' for a,b,v in prefixes.get(left,[]))
        lines.extend(['{',f'const float v={expr};'])
        outputs={}
        for right,path,out,value in entries:outputs.setdefault(out,[]).append((right,path,value))
        for out,rows in outputs.items():
            rhs=_sum(f'{_literal(value)}*w{path}*x{right}' for right,path,value in rows)
            lines.append(f'y{out}+=v*({rhs});')
        lines.append('}')
    if basis_fused:
        offset=0
        for matrix in plan['basis_output']:
            width_out=len(matrix)
            for out,row in enumerate(matrix):
                expr=_basis_sum(row,(f'y{offset+j}' for j in range(width_out)))
                lines.append(f'Y[n*C*{outsize}+{offset}*C+c*{width_out}+{out}]={expr};')
            offset+=width_out
    else:
        for out in range(outsize):lines.append(f'Y[(n*{outsize}+{out})*C+c]=y{out};')
    lines.append('} else {')
    for i in range(size):lines.append(f'float d{i}=0.0f;')
    if basis_fused:
        offset=0
        for matrix in plan['basis_output']:
            width_out=len(matrix)
            for out in range(width_out):
                expr=_basis_sum((row[out] for row in matrix),
                    (f'G[n*C*{outsize}+{offset}*C+c*{width_out}+{j}]' for j in range(width_out)))
                lines.append(f'const float g{offset+out}={expr};')
            offset+=width_out
    else:
        for out in range(outsize):lines.append(f'const float g{out}=G[(n*{outsize}+{out})*C+c];')
    for left,entries in groups.items():
        identity=left>=width
        terms=prefixes.get(left,[])
        expr=f'x{left-width}' if identity else _sum(f'{_literal(v)}*x{a}*x{b}' for a,b,v in terms)
        bv=_sum(f'{_literal(v)}*w{path}*x{right}*g{out}' for right,path,out,v in entries)
        lines.extend(['{',f'const float v={expr};',f'const float bv={bv};'])
        if identity:lines.append(f'd{left-width}+=bv;')
        else:
            derivatives={}
            for a,b,value in terms:
                for i,other in ((a,b),(b,a)):derivatives.setdefault(i,[]).append((other,value))
            for i,rows in derivatives.items():
                rhs=_sum(f'{_literal(value)}*x{other}' for other,value in rows)
                lines.append(f'd{i}+=bv*({rhs});')
        right_rows={}
        for right,path,out,value in entries:
            if right!=size:right_rows.setdefault(right,[]).append((path,out,value))
        for right,rows in right_rows.items():
            rhs=_sum(f'{_literal(value)}*w{path}*g{out}' for path,out,value in rows)
            lines.append(f'd{right}+=v*({rhs});')
        lines.append('}')
    if basis_fused:
        for i,row in enumerate(plan['basis_input']):
            expr=_basis_sum(row,(f'd{j}' for j in range(size)))
            lines.append(f'Y[(n*C+c)*{size}+{i}]={expr};')
    else:
        for i in range(size):lines.append(f'Y[(n*{size}+{i})*C+c]=d{i};')
    lines.extend(['}','}',r'''
void run(torch::Tensor x,torch::Tensor a,torch::Tensor g,torch::Tensor idx,torch::Tensor out,bool backward) {
 const c10::cuda::CUDAGuard guard(x.device());
 TORCH_CHECK(x.is_cuda()&&a.is_cuda()&&out.is_cuda(),"Expected device tensors");
 TORCH_CHECK(x.scalar_type()==at::kFloat&&a.scalar_type()==at::kFloat&&out.scalar_type()==at::kFloat,"Expected FP32");
 TORCH_CHECK(x.is_contiguous()&&a.is_contiguous()&&out.is_contiguous(),"Expected packed tensors");
 const auto N=x.size(0),C=x.size(2);if(N==0||C==0)return;
 auto stream=at::cuda::getCurrentCUDAStream();const auto blocks=(N*C+127)/128;
 if(backward) { circuit<true><<<blocks,128,0,stream>>>(x.data_ptr<float>(),a.data_ptr<float>(),g.data_ptr<float>(),idx.numel()?idx.data_ptr<int64_t>():nullptr,out.data_ptr<float>(),N,C); }
 else { circuit<false><<<blocks,128,0,stream>>>(x.data_ptr<float>(),a.data_ptr<float>(),g.data_ptr<float>(),idx.numel()?idx.data_ptr<int64_t>():nullptr,out.data_ptr<float>(),N,C); }
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''])
    source='\n'.join(lines)
    if basis_fused:source=source.replace('C=x.size(2)','C=x.size(1)')
    return source


def _training_source(plan,banked,mode,hasx,hasa):
    """Keep the inference circuit's node/channel lanes and local CG prefixes.

    Each specialization computes only one requested adjoint. Weight VJPs use
    the existing [path,node,channel] scratch layout, followed by a bounded,
    non-atomic species reduction. No node-sized coefficient bank is gathered.
    """
    pair,final=plan['pair'],plan['final']
    size=plan['input_size'];width=pair['sizes'][3] if pair is not None else 0
    paths=final['sizes'][2];outsize=final['sizes'][3]
    prefixes={};groups={}
    if pair is not None:
        for a,b,_,out,value in pair['entries']:prefixes.setdefault(out,[]).append((a,b,value))
    for left,right,path,out,value in final['entries']:
        groups.setdefault(left,[]).append((right,path,out,value))
    lines=['#include <torch/extension.h>','#include <ATen/cuda/CUDAContext.h>',
           '#include <c10/cuda/CUDAGuard.h>','#include <c10/cuda/CUDAException.h>',
           '__global__ void circuit(const float* X,const float* A,const float* G,const float* H,const float* HA,const int64_t* IDX,float* Y,int64_t N,int64_t C) {',
           'const int64_t row=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;',
           'if(row>=N*C)return;const int64_t n=row/C,c=row%C;',
           f'const int64_t base=({("IDX[n]*C+c" if banked else "c")})*{paths};']
    for i in range(size):lines.append(f'const float x{i}=X[(n*{size}+{i})*C+c];')
    lines.append(f'const float x{size}=1.0f;')
    if hasx:
        for i in range(size):lines.append(f'const float h{i}=H[(n*{size}+{i})*C+c];')
        lines.append(f'const float h{size}=0.0f;')
    if mode!=2:
        for path in range(paths):lines.append(f'const float w{path}=A[base+{path}];')
    if hasa:
        for path in range(paths):lines.append(f'const float a{path}=HA[base+{path}];')
    if mode in (1,2):
        for out in range(outsize):lines.append(f'const float g{out}=G[(n*{outsize}+{out})*C+c];')
    count=size if mode==1 else paths if mode==2 else outsize
    for i in range(count):lines.append(f'float z{i}=0.0f;')
    for left,entries in groups.items():
        identity=left>=width;terms=prefixes.get(left,[])
        value=f'x{left-width}' if identity else _sum(f'{_literal(v)}*x{a}*x{b}' for a,b,v in terms)
        lines.extend(['{',f'const float v={value};'])
        if hasx:
            tangent=f'h{left-width}' if identity else _sum(f'{_literal(v)}*(h{a}*x{b}+x{a}*h{b})' for a,b,v in terms)
            lines.append(f'const float hv={tangent};')
        if mode==1:
            bar=_sum(f'{_literal(v)}*w{path}*x{right}*g{out}' for right,path,out,v in entries)
            lines.append(f'const float bv={bar};')
            hbar=[]
            if hasx:hbar.extend(f'{_literal(v)}*w{path}*h{right}*g{out}' for right,path,out,v in entries)
            if hasa:hbar.extend(f'{_literal(v)}*a{path}*x{right}*g{out}' for right,path,out,v in entries)
            lines.append(f'const float tbv={_sum(hbar)};')
            if identity:
                if left-width<size:lines.append(f'z{left-width}+=tbv;')
            else:
                derivatives={}
                for a,b,coefficient in terms:
                    for i,other in ((a,b),(b,a)):derivatives.setdefault(i,[]).append((other,coefficient))
                for i,rows in derivatives.items():
                    pv=_sum(f'{_literal(v)}*x{other}' for other,v in rows)
                    rhs=[f'tbv*({pv})']
                    if hasx:rhs.append('bv*('+_sum(f'{_literal(v)}*h{other}' for other,v in rows)+')')
                    lines.append(f'z{i}+={_sum(rhs)};')
            right_rows={}
            for right,path,out,v in entries:
                if right!=size:right_rows.setdefault(right,[]).append((path,out,v))
            for right,rows in right_rows.items():
                rhs=[]
                if hasx:rhs.append('hv*('+_sum(f'{_literal(v)}*w{path}*g{out}' for path,out,v in rows)+')')
                if hasa:rhs.append('v*('+_sum(f'{_literal(v)}*a{path}*g{out}' for path,out,v in rows)+')')
                lines.append(f'z{right}+={_sum(rhs)};')
        elif mode==2:
            path_rows={}
            for right,path,out,v in entries:path_rows.setdefault(path,[]).append((right,out,v))
            for path,rows in path_rows.items():
                rhs=_sum(f'{_literal(v)}*g{out}*x{right}' for right,out,v in rows)
                if hasx:
                    hrhs=_sum(f'{_literal(v)}*g{out}*h{right}' for right,out,v in rows)
                    lines.append(f'z{path}+=hv*({rhs})+v*({hrhs});')
                else:lines.append(f'z{path}+=v*({rhs});')
        else:
            output_rows={}
            for right,path,out,v in entries:output_rows.setdefault(out,[]).append((right,path,v))
            for out,rows in output_rows.items():
                rhs=[]
                if hasx:
                    rhs.append('hv*('+_sum(f'{_literal(v)}*w{path}*x{right}' for right,path,v in rows)+')')
                    rhs.append('v*('+_sum(f'{_literal(v)}*w{path}*h{right}' for right,path,v in rows)+')')
                if hasa:rhs.append('v*('+_sum(f'{_literal(v)}*a{path}*x{right}' for right,path,v in rows)+')')
                lines.append(f'z{out}+={_sum(rhs)};')
        lines.append('}')
    for i in range(count):
        offset=f'({i}*N+n)*C+c' if mode==2 else f'(n*{count}+{i})*C+c'
        lines.append(f'Y[{offset}]=z{i};')
    lines.extend(['}',r'''
void run(torch::Tensor x,torch::Tensor a,torch::Tensor g,torch::Tensor h,torch::Tensor ha,torch::Tensor idx,torch::Tensor out) {
 const c10::cuda::CUDAGuard guard(x.device());
 for(const auto& t:{x,a,g,h,ha,out}) TORCH_CHECK(t.is_cuda()&&t.device()==x.device()&&t.scalar_type()==at::kFloat&&t.is_contiguous(),"Expected packed FP32 circuit operands");
 const auto N=x.size(0),C=x.size(2);if(N==0||C==0)return;
 auto stream=at::cuda::getCurrentCUDAStream();const auto blocks=(N*C+127)/128;
 circuit<<<blocks,128,0,stream>>>(x.data_ptr<float>(),a.data_ptr<float>(),g.data_ptr<float>(),h.data_ptr<float>(),ha.data_ptr<float>(),idx.numel()?idx.data_ptr<int64_t>():nullptr,out.data_ptr<float>(),N,C);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void reduce_weights(const float* V,const int64_t* IDX,float* O,int64_t N,int64_t C,int64_t P,int64_t chunks) {
 const int64_t pc=static_cast<int64_t>(blockIdx.y)*32+threadIdx.x;
 const int64_t chunk=blockIdx.x,s=blockIdx.z;
 float sum=0.0f;
 if(pc<P*C) for(int offset=threadIdx.y;offset<256;offset+=8) {
   const int64_t n=chunk*256+offset;
   if(n<N&&IDX[n]==s)sum+=V[(pc/C*N+n)*C+pc%C];
 }
 __shared__ float values[8][32];values[threadIdx.y][threadIdx.x]=sum;
 __syncthreads();
 if(threadIdx.y==0&&pc<P*C) {
   float total=0.0f;for(int row=0;row<8;++row)total+=values[row][threadIdx.x];
   O[(s*chunks+chunk)*(P*C)+pc]=total;
 }
}

void reduce(torch::Tensor v,torch::Tensor idx,torch::Tensor out) {
 const c10::cuda::CUDAGuard guard(v.device());
 const auto P=v.size(0),N=v.size(1),C=v.size(2),chunks=out.size(1),S=out.size(0);
 if(!out.numel())return;
 auto stream=at::cuda::getCurrentCUDAStream();
 reduce_weights<<<dim3(chunks,(P*C+31)/32,S),dim3(32,8),0,stream>>>(v.data_ptr<float>(),idx.data_ptr<int64_t>(),out.data_ptr<float>(),N,C,P,chunks);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''])
    return '\n'.join(lines)


def _training_raw(x,a,plan,mode,g,hx,ha,idx,packed_input):
    banked=idx is not None
    signature=('training',banked,mode,hx is not None,ha is not None)
    cache=plan.setdefault('hip_circuit_extensions',{})
    if signature not in cache:
        source=_training_source(plan,banked,mode,hx is not None,ha is not None)
        key=hashlib.sha256(source.encode()).hexdigest()[:16]
        if key not in _EXTENSIONS:
            from ictc_o3.backends.k100.native_jit import load_inline
            declarations='#include <torch/extension.h>\nvoid run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);\nvoid reduce(torch::Tensor,torch::Tensor,torch::Tensor);'
            _EXTENSIONS[key]=load_inline(name='ictc_hip_circuit_'+key,
                cpp_sources=declarations,cuda_sources=source,functions=['run','reduce'],
                extra_cflags=['-O3'],extra_cuda_cflags=['-O3'])
        cache[signature]=_EXTENSIONS[key]
    n=x.shape[0];c=x.shape[2] if packed_input else x.shape[1]
    i=plan['input_size'];p=plan['final']['sizes'][2];m=plan['final']['sizes'][3]
    packed=x if packed_input else x.transpose(1,2).contiguous()
    gp=packed if g is None else g if packed_input else g.transpose(1,2).contiguous()
    hp=packed if hx is None else hx if packed_input else hx.transpose(1,2).contiguous()
    indices=torch.empty(0,device=x.device,dtype=torch.int64) if idx is None else idx.contiguous()
    out=x.new_empty((p,n,c) if mode==2 else (n,i if mode==1 else m,c))
    cache[signature].run(packed,a.contiguous(),gp,hp,a if ha is None else ha.contiguous(),indices,out)
    if mode==2:
        if banked:
            partial=out.new_empty(a.shape[0],(n+255)//256,p,c)
            cache[signature].reduce(out,indices,partial)
            return partial.sum(1).transpose(1,2).contiguous()
        return out.sum(1).T.contiguous()
    return out if packed_input else out.transpose(1,2).contiguous()


def raw_basis(x,a,index,plan,mode=0,grad=None):
    """Frozen inference circuit in the original feature and output layouts."""
    if mode not in (0,1):raise ValueError('Unknown fused basis circuit mode')
    n,c,_=x.shape;m=plan['final']['sizes'][3]
    out=torch.empty_like(x) if mode==1 else x.new_empty(n,c*m)
    if not out.numel():return out
    signature=('basis',True)
    cache=plan.setdefault('hip_circuit_extensions',{})
    if signature not in cache:
        source=_source(plan,True,basis_fused=True)
        key=hashlib.sha256(source.encode()).hexdigest()[:16]
        if key not in _EXTENSIONS:
            from ictc_o3.backends.k100.native_jit import load_inline
            _EXTENSIONS[key]=load_inline(name='ictc_hip_circuit_'+key,
                cpp_sources='#include <torch/extension.h>\nvoid run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,bool);',
                cuda_sources=source,functions=['run'],extra_cflags=['-O3'],extra_cuda_cflags=['-O3'])
        cache[signature]=_EXTENSIONS[key]
    cache[signature].run(x,a,x if grad is None else grad,index,out,mode==1)
    return out


def raw(x,a,plan,mode,g=None,idx=None,*,hx=None,ha=None,packed_input=False):
    if mode not in (0,1,2,3):raise ValueError('Unknown HIP circuit mode')
    if mode in (2,3) or hx is not None or ha is not None:
        return _training_raw(x,a,plan,mode,g,hx,ha,idx,packed_input)
    cache=plan.setdefault('hip_circuit_extensions',{})
    banked=idx is not None
    if banked not in cache:
        source=_source(plan,banked)
        key=hashlib.sha256(source.encode()).hexdigest()[:16]
        if key not in _EXTENSIONS:
            from ictc_o3.backends.k100.native_jit import load_inline
            _EXTENSIONS[key]=load_inline(name='ictc_hip_circuit_'+key,
                cpp_sources='#include <torch/extension.h>\nvoid run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,bool);',
                cuda_sources=source,functions=['run'],extra_cflags=['-O3'],extra_cuda_cflags=['-O3'])
        cache[banked]=_EXTENSIONS[key]
    n=x.shape[0];i=x.shape[1] if packed_input else x.shape[2]
    c=x.shape[2] if packed_input else x.shape[1];m=plan['final']['sizes'][3]
    packed=x if packed_input else x.transpose(1,2).contiguous()
    out=x.new_empty(n,i if mode==1 else m,c)
    indices=torch.empty(0,device=x.device,dtype=torch.int64) if idx is None else idx.contiguous()
    gp=packed if g is None else g if packed_input else g.transpose(1,2).contiguous()
    cache[banked].run(packed,a.contiguous(),gp,indices,out,mode==1)
    return out if packed_input else out.transpose(1,2).contiguous()
