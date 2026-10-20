# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Output-degree partitions of the exact independent FP32 CG circuit.

Only immutable final-entry metadata is partitioned. Every projected learned
weight stays live. Each degree uses the established first/mixed circuit
adjoints, reducing per-thread live path weights and weight-cotangent scratch.
"""
import torch
from .factorized_circuit import factorized_circuit


def partitions(plan,device):
    previous=plan.get('degree_partitions')
    if previous is not None:return previous
    left,right,_,_=plan['final']['sizes']
    result=[];offset=0
    for width in plan['sizes']:
        rows=[row for row in plan['final']['entries'] if offset<=row[3]<offset+width]
        ids=sorted({row[2] for row in rows})
        lookup={path:i for i,path in enumerate(ids)}
        entries=tuple((a,b,lookup[p],out-offset,value) for a,b,p,out,value in rows)
        part={'pair':plan['pair'],'input_size':plan['input_size'],
              'compact_adjoints':plan.get('compact_adjoints',False),
              'final':{'sizes':(left,right,len(ids),width),'entries':entries}}
        index=torch.tensor(ids,device=device,dtype=torch.int64)
        result.append((part,index))
        offset+=width
    plan['degree_partitions']=tuple(result)
    return plan['degree_partitions']


def partitioned_circuit(x,a,plan):
    values=[]
    for part,index in partitions(plan,x.device):
        values.append(factorized_circuit(x,a.index_select(1,index),part))
    return torch.cat(values,dim=-1)


def compact_plan(plan):
    previous=plan.get('compact_adjoint_plan')
    if previous is not None:return previous
    variant={k:v for k,v in plan.items() if k not in ('circuit_signature','circuit_resources','degree_partitions','compact_adjoint_plan')}
    variant['compact_adjoints']=True
    plan['compact_adjoint_plan']=variant
    return variant
