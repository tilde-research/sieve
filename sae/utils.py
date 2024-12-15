import os
import json
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from typing import List, Optional, Dict
from dataclasses import dataclass
import mmap

@dataclass
class SaeConfig:
    """Configuration for SAE inference - only essential parameters"""
    d_model: int
    expansion_factor: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "torch.bfloat16"
    apply_activation_fn: bool = True
    activation_fn_names: List[str] = None
    activation_fn_kwargs: List[dict] = None
    eval_activation_idx: int = 0

    @classmethod
    def from_dict(cls, config_dict: dict) -> "SaeConfig":
        essential_params = {
            'd_model', 'expansion_factor', 'device', 'dtype',
            'apply_activation_fn',
            'activation_fn_names', 'activation_fn_kwargs',
            'eval_activation_idx'
        }
        filtered_dict = {k: v for k, v in config_dict.items() if k in essential_params}
        return cls(**filtered_dict)
