# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Independent path-grouped exact FP32 edge messages and feature/gate VJPs."""
from functools import lru_cache
import importlib.util
import itertools
import os
from pathlib import Path
import sys
import torch
from .monomial_operator import triton

_SERIAL=itertools.count()


def _sum(values):return '+'.join(f'({v})' for v in values) if values else '0.0'


@lru_cache(maxsize=128)
def _kernels(specs,xdim,odim,c,paths,hdim,bdim,rows):
    columns=[[] for _ in range(bdim)]
    for h,items in enumerate(rows):
        for col,coefficient in items:columns[col].append((h,coefficient))
    groups={}
    for row in specs:groups.setdefault((row[2],row[1]),[]).append(row)
    source=['import triton','import triton.language as tl']
    names=[]
    for kind in ('forward','adjoint'):
        for group,((xb,ni),items) in enumerate(groups.items()):
            name=f'{kind}_{group}';names.append(name)
            source += ['@triton.jit',f'def {name}(X,SH,W,G,S,D,M,V,O,DX,DG,E,MASKED:tl.constexpr,UNIT:tl.constexpr,NEEDX:tl.constexpr,NEEDG:tl.constexpr,BLOCK:tl.constexpr):',
                '    index=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)',
                f'    edge=index//{c};channel=index%{c};active=edge<E',
                '    src=tl.load(S+edge,active,0);dst=tl.load(D+edge,active,0)',
                '    if MASKED:','        mask=tl.load(M+edge,active,0.)','    else:','        mask=1.0']
            used=sorted({h for _,ni,xb,bb,bs,ob,m in items for i in range(ni) for k in range(m) for h,_ in columns[bb+i*bs+k]})
            for h in used:source.append(f'    h{h}=tl.load(SH+edge*{hdim}+{h},active,0.)')
            for i in range(ni):
                if kind=='forward':source.append(f'    x{i}=tl.load(X+src*{xdim}+{xb+i*c}+channel,active,0.)')
                else:
                    source += ['    if NEEDG:',f'        x{i}=tl.load(X+src*{xdim}+{xb+i*c}+channel,active,0.)',f'    dx{i}=tl.full((BLOCK,),0,tl.float32)']
            for path,ni,xb,bb,bs,ob,m in items:
                source += ['    if UNIT:','        w=1.0','    else:',f'        w=tl.load(W+{path*c}+channel,active,0.)']
                if kind=='forward':source.append(f'    gate=tl.load(G+edge*{paths*c}+{path*c}+channel,active,0.)*w*mask')
                else:
                    source += ['    if NEEDX:',f'        gate=tl.load(G+edge*{paths*c}+{path*c}+channel,active,0.)*w*mask',
                        '    if NEEDG:','        dg=tl.full((BLOCK,),0,tl.float32)']
                    for k in range(m):source.append(f'    v{k}=tl.load(V+dst*{odim}+{ob+k*c}+channel,active,0.)')
                if kind=='forward':
                    for k in range(m):
                        terms=[]
                        for i in range(ni):
                            coefficients=columns[bb+i*bs+k]
                            if coefficients:terms.append(f'x{i}*('+_sum(f'{float(a)!r}*h{h}' for h,a in coefficients)+')')
                        source += [f'    value=gate*('+_sum(terms)+')',f'    tl.atomic_add(O+dst*{odim}+{ob+k*c}+channel,value,active,sem="relaxed")']
                else:
                    for i in range(ni):
                        terms=[]
                        for k in range(m):
                            coefficients=columns[bb+i*bs+k]
                            if coefficients:terms.append(f'v{k}*('+_sum(f'{float(a)!r}*h{h}' for h,a in coefficients)+')')
                        source += [f'    z{i}='+_sum(terms),'    if NEEDX:',f'        dx{i}+=gate*z{i}',
                            '    if NEEDG:',f'        dg+=x{i}*z{i}']
                    source += ['    if NEEDG:',f'        tl.store(DG+edge*{paths*c}+{path*c}+channel,dg*w*mask,active)']
            if kind=='adjoint':
                for i in range(ni):source += ['    if NEEDX:',f'        tl.atomic_add(DX+src*{xdim}+{xb+i*c}+channel,dx{i},active,sem="relaxed")']
    root=Path(os.environ.get('TRITON_CACHE_DIR',str(Path.home()/'.triton/cache')))/'cartesian_message_groups'
    root.mkdir(parents=True,exist_ok=True);name=f'ictc_message_groups_{os.getpid()}_{next(_SERIAL)}';path=root/f'{name}.py'
    path.write_text('\n'.join(source)+'\n');spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    split=len(names)//2
    return [getattr(module,k) for k in names[:split]],[getattr(module,k) for k in names[split:]]


def _launch(x,sh,w,g,src,dst,mask,v,out,dx,dg,p,projection,kind,needed):
    if not src.numel():return
    if p.get('generated_layout')=='canonical':raise ValueError('Grouped Triton messages require packed kernel inputs')
    kernels=_kernels(p['path_specs'],p['xdim'],p['odim'],p['c'],p['p'],projection['hdim'],p['bdim'],projection['static_rows'])[kind]
    block=int(p.get('triton_message_block',128))
    if block not in (64,128,256):raise ValueError('Grouped message block must be 64,128 or256')
    for kernel in kernels:
        kernel[(triton.cdiv(src.numel()*p['c'],block),)](x,sh,w,g,src,dst,mask,v,out,dx,dg,src.numel(),
            bool(mask.numel()),bool(p.get('unit_weights',False)),*needed,block,num_warps=4,enable_fp_fusion=True)


def forward(x,sh,w,g,src,dst,mask,out,p,projection):
    _launch(x,sh,w,g,src,dst,mask,x,out,x,g,p,projection,0,(False,False))


def adjoint(x,sh,w,g,src,dst,mask,v,dx,dg,p,projection,needed):
    _launch(x,sh,w,g,src,dst,mask,v,v,x if dx is None else dx,g if dg is None else dg,p,projection,1,needed)
