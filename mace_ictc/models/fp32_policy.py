# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The HP package uses ordinary IEEE FP32, including library GEMMs."""
import torch


def strict_fp32():
    """Disable TF32 globally; never cast tensors or change trainability."""
    if torch.get_float32_matmul_precision()!='highest' or torch.backends.cuda.matmul.allow_tf32:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False
    if torch.backends.cudnn.allow_tf32:torch.backends.cudnn.allow_tf32=False
