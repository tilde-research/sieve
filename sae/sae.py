import torch
from torch import nn
import pickle
import json
import os
from typing import Tuple, Union
from huggingface_hub import hf_hub_download
from torch import Tensor
from .utils import SaeConfig

""" This logic is taken from Eleuther's SAE repo. """
# Fallback implementation of SAE decoder
def eager_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
    buf = top_acts.new_zeros(top_acts.shape[:-1] + (W_dec.shape[-1],))
    acts = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
    return acts @ W_dec.mT


# Triton implementation of SAE decoder
def triton_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
    return TritonDecoder.apply(top_indices, top_acts, W_dec)


try:
    from .kernels import TritonDecoder
except ImportError:
    decoder_impl = eager_decode
    print("Triton not installed, using eager implementation of SAE decoder.")
else:
    if os.environ.get("SAE_DISABLE_TRITON") == "1":
        print("Triton disabled, using eager implementation of SAE decoder.")
        decoder_impl = eager_decode
    else:
        decoder_impl = triton_decode


class TopK(nn.Module):
    __is_sparse__ = True
    """Top-k activation function"""
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def sparse_forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = nn.ReLU()(x)
        return torch.topk(x, self.k, dim=-1, sorted=False)
    
    def dense_forward(self, x: Tensor) -> Tensor:
        acts, indices = self.sparse_forward(x)
        return acts.scatter_(dim=-1, index=indices, src=acts)

    def forward(self, x: Tensor) -> Union[Tuple[Tensor, Tensor], Tensor]:
        if self.__is_sparse__:
            return self.sparse_forward(x)
        else:
            return self.dense_forward(x)
        
    def __repr__(self):
        return f"TopK(k={self.k})"

class Sae(nn.Module):
    """Streamlined Sparse Autoencoder for inference only"""
    
    def __init__(self, config: SaeConfig):
        super().__init__()
        self.cfg = config
        self.d_in = config.d_model
        self.d_sae = config.d_model * config.expansion_factor
        
        # Core parameters
        self.W_enc_DF = nn.Parameter(torch.empty(self.d_in, self.d_sae))
        self.b_enc_F = nn.Parameter(torch.zeros(self.d_sae))
        self.W_dec_FD = nn.Parameter(torch.empty(self.d_sae, self.d_in))
        self.b_dec_D = nn.Parameter(torch.zeros(self.d_in))
        
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype.split(".")[1])
        self.activation_fns = None
        self.to(self.device, self.dtype)
        self.eval()

    @torch.no_grad()
    def encode(self, x: torch.Tensor, activation_fn_idx: int = None) -> torch.Tensor:
        if activation_fn_idx is None:
            activation_fn_idx = self.cfg.eval_activation_idx



        pre_acts = torch.matmul(x, self.W_enc_DF) + self.b_enc_F
        acts = self.activation_fns[activation_fn_idx](pre_acts)
        return acts

    @torch.no_grad()
    def decode(self, features: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], eager: bool = False) -> torch.Tensor:
        if isinstance(features, tuple): # Sparse feats
            top_indices, top_acts = features
            if eager or top_indices.shape[-1] >= 512: # Heuristic for when triton kernel is slower than eager decoding
                y = eager_decode(top_indices, top_acts, self.W_dec_FD.mT)
            else:
                y = decoder_impl(top_indices, top_acts, self.W_dec_FD.mT)
        else:
            y = torch.matmul(features, self.W_dec_FD)
        y = y + self.b_dec_D
        return y

    @torch.no_grad()
    def forward(self, x: torch.Tensor, activation_fn_idx: int = None) -> torch.Tensor:
        if activation_fn_idx is None:
            activation_fn_idx = self.cfg.eval_activation_idx
        initial_shape = x.shape
        x = x.reshape(-1, self.d_in)
        f = self.encode(x, activation_fn_idx)
        return self.decode(f).reshape(initial_shape), f

    @classmethod
    def from_pretrained(cls, repo_id: str, cache_dir: str = None, layer_idx: int = 12, token=None) -> "Sae":
        if cache_dir is None:
            cache_dir = os.path.expanduser(f"~/.cache/huggingface/{repo_id}")
        os.makedirs(cache_dir, exist_ok=True)

        config_file = hf_hub_download(repo_id=repo_id, filename="config.json", cache_dir=cache_dir, token=token)
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        assert "sae_cfg" in config_dict, "config.json must contain 'sae_cfg' key"
        config = SaeConfig.from_dict(config_dict["sae_cfg"])
        model = cls(config)

        weight_file = f"layer_{layer_idx}.pt"
        weights_path = hf_hub_download(repo_id=repo_id, filename=weight_file, cache_dir=cache_dir, token=token)
        
        with open(weights_path, "rb") as f:
            state_dict = pickle.load(f)
        
        model.load_state_dict(state_dict)

        model.activation_fns = [TopK(k=config.activation_fn_kwargs[i]["k"]) for i in range(len(config.activation_fn_kwargs))]

        return model

    def __repr__(self):
        return f"Sae(d_in={self.d_in}, d_sae={self.d_sae})" 
    
    @property
    def W_enc(self):
        return self.W_enc_DF
    
    @property
    def W_dec(self):
        return self.W_dec_FD
    
    @property
    def b_enc(self):
        return self.b_enc_F
    
    @property
    def b_dec(self):
        return self.b_dec_D