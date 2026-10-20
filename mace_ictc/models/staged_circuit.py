# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Exact shared quadratic prefixes followed by degree-local weighted circuits.

The high-degree cubic circuit otherwise expands each quadratic derivative in
every final coupling. Materializing a shared prefix separates that polynomial
from the bilinear final stage and keeps both first and mixed adjoints live.
"""
import torch

from .partitioned_circuit import partitioned_circuit
from .sparse_weighted_product import sparse_quadratic


def staged_plan(plan):
    previous=plan.get("staged_plan")
    if previous is not None:return previous
    width=plan["width"];size=plan["input_size"]
    _,_,paths,outputs=plan["final"]["sizes"]
    entries=tuple((left,width+right,path,out,value)
                  for left,right,path,out,value in plan["final"]["entries"])
    variant={"pair":None,"input_size":width+size,"sizes":plan["sizes"],
             "compact_adjoints":True,
             "final":{"sizes":(width+size,width+size+1,paths,outputs),"entries":entries}}
    plan["staged_plan"]=variant
    return variant


def staged_circuit(x,a,plan):
    pairs=(sparse_quadratic(x,plan["ones"],plan["pair"],plan["pair_adjoint"])
           if plan["pair"] is not None else x[:,:,:0])
    return partitioned_circuit(torch.cat((pairs,x),-1),a,staged_plan(plan))
