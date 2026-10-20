# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Bounded generated quadratic CG blocks with exact mixed adjoints.

Each fixed (l1,l2,lout) prefix owns at most 2*lout+1 outputs. Splitting those
constant blocks avoids the generic sparse map's largest padded reduction and
the full cubic circuit's large generated program. Every coefficient is kept.
"""
import torch

from .factorized_circuit import factorized_circuit


def bounded_quadratic(x,plan):
    cached=plan.get("bounded_pair_plans")
    if cached is None:
        size=plan["input_size"]
        cached=[]
        for offset,width in plan["pair_blocks"]:
            entries=tuple((a,b,0,out-offset,value) for a,b,_,out,value in plan["pair"]["entries"]
                          if offset<=out<offset+width)
            cached.append({"pair":None,"input_size":size,
                           "final":{"sizes":(size,size+1,1,width),"entries":entries}})
        plan["bounded_pair_plans"]=tuple(cached)
    return torch.cat([factorized_circuit(x,plan["ones"],part) for part in cached],-1)
