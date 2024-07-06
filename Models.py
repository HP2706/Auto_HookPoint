
from abc import ABC, abstractmethod
from typing_extensions import Self
import bitsandbytes as bnb
from torch import nn, Tensor
import torch
import inspect
from typing import List, Optional, Type, Union, Callable, Protocol, Literal, cast, overload, TypeVar
from pydantic import BaseModel, model_validator
from jaxtyping import Float, Int
from schedulefree import AdamWScheduleFree
from typing import Callable, Optional, Union
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped, Int, Bool
from typing import OrderedDict
try:
    from mamba_ssm.ops.triton.layer_norm import layer_norm_fn, rms_norm_fn
except ImportError:
    layer_norm_fn, rms_norm_fn = None, None

from os import remove
import math
from pydantic import BaseModel, Field, model_validator
from typing import Union, Protocol, Optional, Literal
import torch.nn as nn
import torch
from typing_extensions import Self

import os
import math 
import warnings
import torch
from typing import Optional, Union
from pydantic import BaseModel, model_validator
import numpy as np
from torch import Tensor
from typing import Tuple

class LRConfig(BaseModel):
    warmup_steps : Optional[int]
    max_steps : int
    min_lr : Optional[float]
    max_lr : Optional[float]
    schedule_free : bool = False

    @model_validator(mode='after')
    def validate_lr_config(cls, v):
        messages = []
        errors = []
        if v.schedule_free:
            if v.warmup_steps is not None:
                messages.append("warmup_steps should not be defined when schedule_free is True")
            if v.min_lr is not None:
                messages.append("min_lr should not be defined when schedule_free is True")
            if v.max_lr is not None:
                messages.append("max_lr should not be defined when schedule_free is True")
        else:
            if v.warmup_steps is None:
                errors.append("warmup_steps must be defined when schedule_free is False")
            if v.min_lr is None:
                errors.append("min_lr must be defined when schedule_free is False")
            if v.max_lr is None:
                errors.append("max_lr must be defined when schedule_free is False")
        if errors:
            raise ValueError(f"Invalid LRConfig: {v.model_dump()} {errors}")
        if messages:
            warnings.warn(f'''
                Warning the schedulefree optimizer will not use 
                these values: {messages}. 
            ''')

        return v


class LRScheduler:
    def __init__(
        self, 
        config : LRConfig
    ) -> None:
        self.warmup_steps = config.warmup_steps
        self.max_steps = config.max_steps
        self.min_lr = config.min_lr
        self.max_lr = config.max_lr

    def get_lr(self, it : int) -> float:
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_steps:
            return self.max_lr * (it+1) / self.warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.max_steps:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return self.min_lr + coeff * (self.max_lr - self.min_lr)


def init_log_file(log_dir = 'log_dir', log_file = 'log.txt')-> None:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, log_file)
    with open(log_file, "w") as f: # open for writing to clear the file
        pass

def get_device() -> Union[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device('cuda')
    #elif torch.backends.mps.is_available():
    #    return torch.device('mps')
    else:
        return 'cpu'

def text_to_bytes(texts: list[str]) -> torch.Tensor:
    '''converts text to bytes and returns [batch, seq_len] tensor 
    and pads to max_seq_len in batch with zeros'''
    n_bytes = torch.nn.utils.rnn.pad_sequence(
        [
            torch.Tensor(list(text.encode('utf-8'))
            ).to(dtype=torch.long) for text in texts
        ],
        batch_first=True
    )
    return n_bytes

def bytes_to_text(byte_tensor: torch.Tensor) -> list[str]:
    '''converts bytes in torch.Tensor to text'''
    texts = []
    byte_tensor = byte_tensor.to(dtype=torch.uint8)
    for i in range(byte_tensor.size(0)): # iter over batch
        text = byte_tensor[i][byte_tensor[i] != 0].tolist()  # Skip null bytes
        texts.append(bytes(text).decode('utf-8', errors='replace'))
    return texts

def img_to_bytes(images: np.ndarray) -> Tensor:
    '''converts a list of images to bytes, images are shape [batch, channels, height, width]'''
    return torch.from_numpy(images.astype(np.uint8)).reshape(images.shape[0], -1).to(dtype=torch.long)  # batch first

def bytes_to_img(bytes: torch.Tensor, image_shape: Tuple[int, ...]) -> np.ndarray:
    '''converts a list of bytes to images, images are shape [batch, channels, height, width]'''
    bytes = bytes.to(dtype=torch.uint8)
    return bytes.numpy().reshape((-1,) + image_shape)  # batch first

def count_non_embedding_params(model):
    def is_embedding_param(module, param):
        out = isinstance(module, torch.nn.Embedding)
        return out

    non_embedding_params = []
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not is_embedding_param(module, param):
                non_embedding_params.append(param)
    return sum(param.numel() for param in non_embedding_params)

from torch.nn import functional as F
def pad_sequence(tokens: torch.Tensor, patch_size: int, pad_id : int) -> torch.Tensor:
    tokens_to_pad = tokens.size(1) % patch_size
    tokens = F.pad(tokens, (0, tokens_to_pad), "constant", pad_id)
    return tokens



class ModelConfig(BaseModel):
    n_ctx : int = 1024 # block_size
    vocab_size : int = 50257 
    use_adam_8_bit : bool = Field(default=False, description="""
        if true, use AdamW8bit optimizer
    """)
    is_training : bool
    eps : float = 1e-5
    nonlin : nn.Module = nn.GELU(approximate='tanh')
    class Config:
        arbitrary_types_allowed = True

class BaseTransformerConfig(ModelConfig):
    d_model : int
    d_mult : int = 4 # residual stream = d_mult*d_model
    num_heads : int
    eps : float = 1e-5
    n_layers : int
    is_causal : bool = True

    @property
    def d_head(self) -> int:
        return self.d_model // self.num_heads

    class Config:
        arbitrary_types_allowed = True


class HasConfig(Protocol):
    config: Union[ModelConfig, dict]

class MoDConfig(BaseTransformerConfig):
    capacity : float = 0.125 #12.5% as in mixture-of-depths paper
    every_other : bool = Field(default=False, description="""
        if true, only every second block is a Mod and the rest is regular transformerblock
    """)

class VanillaConfig(BaseTransformerConfig):
    pass

class MambaConfig(ModelConfig):
    d_model: int
    d_state: int
    n_layers: int
    pad_token_id: int = Field(default=0, description="Padding token id.")
    bos_token_id: int = Field(default=0, description="The id of the beginning of sentence token in the vocabulary.")
    fused_add_norm : bool = Field(default=False, description="Whether or not to fuse add and norm")
    eos_token_id: int = Field(default=0, description="The id of the end of sentence token in the vocabulary.")
    expand: int = Field(default=2, description="Expanding factor used to determine the intermediate size.")
    d_conv: int = Field(default=4, description="Size of the convolution kernel.")
    use_bias: bool = Field(default=False, description="Whether or not to use bias in [\"in_proj\", \"out_proj\"] of the mixer block")
    use_conv_bias: bool = Field(default=True, description="Whether or not to use bias in the convolution layer of the mixer block.")
    hidden_act: str = Field(default="silu", description="The non-linear activation function in the decoder.")
    initializer_range: float = Field(default=0.1, description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.")
    residual_in_fp32: bool = Field(default=True, description="Whether or not residuals should be in `float32`.")
    time_step_rank: Union[int, Literal["auto"]] = Field(default="auto", description="Rank of the discretization projection matrix.")
    time_step_scale: float = Field(default=1.0, description="Scale used to scale `dt_proj.bias`.")
    time_step_min: float = Field(default=0.001, description="Minimum `time_step` used to bound `dt_proj.bias`.")
    time_step_max: float = Field(default=0.1, description="Maximum `time_step` used to bound `dt_proj.bias`.")
    time_step_init_scheme: Literal["random", "uniform"] = Field(default="random", description="Init scheme used for `dt_proj.weight`.")
    time_step_floor: float = Field(default=1e-4, description="Minimum clamping value of the `dt_proj.bias` layer initialization.")
    rescale_prenorm_residual: bool = Field(default=False, description="Whether or not to rescale `out_proj` weights when initializing.")

    @property
    def d_intermediate(self) -> int:
        return int(self.expand * self.d_model)

    @property
    def time_step_rank_value(self) -> int:
        return math.ceil(self.d_model / 16) if self.time_step_rank == "auto" else self.time_step_rank

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        if self.time_step_rank == "auto":
            self.time_step_rank = math.ceil(self.d_model / 16)

class MoEConfig(BaseTransformerConfig):
    num_experts : int
    top_k : int
    jitter_noise : Optional[float] = None
    alpha : Optional[float] = Field(
        default=None, 
        description="this is the aux loss weight"
    )
    alphas : Optional[list[float]] = Field(
        default=None,
        description='''
        this is the aux loss weight but 
        where a different weight can be 
        applied to each layer
        '''
    )

    #beta/betas is inspired by router z loss coeff in paper: https://arxiv.org/pdf/2202.08906 
    beta : Optional[float] = Field(default=None, description="""
        the router z loss, if beta is 
        chosen the same loss is applied 
        to each layer
    """)
    betas : Optional[list[float]] = Field(default=None, description="""
        the router z loss, if betas is 
        chosen a different loss is applied 
        to each layer
    """)
    gate_type : Literal["mixtral"] = "mixtral"
    gate_norm_factor : Optional[float] = None

    @model_validator(mode='before')
    @classmethod
    def validate_alphas_betas(cls, values):
        n_layers = values.get('n_layers')
        if values.get('alphas') is not None and values.get('betas') is not None:
            assert len(values.get('alphas')) == n_layers and len(values.get('betas')) == n_layers

        if not values.get('alpha') and not values.get('alphas'):
            raise ValueError("Either alpha or alphas must be set.")
        if not values.get('beta') and not values.get('betas'):
            raise ValueError("Either beta or betas must be set.")

        #disambiguate
        if values.get('alpha') and values.get('alphas'):
            raise ValueError("Only one of alpha or alphas can be set, not both.")
        if values.get('beta') and values.get('betas'):
            raise ValueError("Only one of beta or betas can be set, not both.")
        return values
    
class MegaByteConfig(ModelConfig):
    patch_size : int = 16
    d_local : int 
    d_global_pre_patch : int 
    n_layers_d_global : int
    n_layers_d_local : int
    local_n_heads : int
    global_n_heads : int  
    d_mult : int  # residual stream = d_mult*d_local/d_global
    is_causal : bool = True
    pad_id : int = 257
    eos_id : int = 258
    vocab_size : int = 256+2 #2 with  eos and pad

    @property 
    def local_d_head(self) -> int:
        assert self.d_local % self.local_n_heads == 0
        return self.d_local // self.local_n_heads

    @property
    def global_d_head(self) -> int:
        assert self.d_global_pre_patch % self.global_n_heads == 0
        return self.d_global_pre_patch // self.global_n_heads


class LLama3Config(BaseTransformerConfig):
    n_kv_heads : Optional[int] = None
    with_linear_bias : bool = False
    max_emb_len : int #for ROPE

CONFIG_MAP = {
    ModelConfig.__name__ : ModelConfig, 
    VanillaConfig.__name__ : VanillaConfig,
    MoDConfig.__name__ : MoDConfig,
    MoEConfig.__name__ : MoEConfig,
    MambaConfig.__name__ : MambaConfig,
    BaseTransformerConfig.__name__ : BaseTransformerConfig,
    LLama3Config.__name__ : LLama3Config,
    MegaByteConfig.__name__ : MegaByteConfig
}



class MLP(nn.Module):
    def __init__(self, cfg : BaseTransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_model*cfg.d_mult)
        self.fc2 = nn.Linear(cfg.d_model*cfg.d_mult, cfg.d_model)
        self.fc2.NANOGPT_SCALE_INIT = 1 #type: ignore
        self.nonlin = cfg.nonlin

    ##@jaxtyped(typechecker=beartype) TODO torch.compile doesn't work with jaxtyped decorators
    def forward(
        self, 
        x : Union[
            Float[Tensor, "batch sequence_len d_model"],
            Float[Tensor, "batch d_model"]
        ],
    )-> Union[
        Float[Tensor, "batch sequence_len d_model"],
        Float[Tensor, "batch d_model"]
    ]:
        x = self.fc1(x)
        x = self.nonlin(x)
        x = self.fc2(x)
        return x

class UnEmbedding(nn.Module):
    def __init__(self, cfg: BaseTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Linear(cfg.d_model, cfg.vocab_size, bias = False)

    #@jaxtyped(typechecker=beartype) TODO torch.compile doesn't work with jaxtyped decorators
    def forward(
        self, 
        x : Float[Tensor, "batch sequence_len d_model"],
    )-> Float[Tensor, "batch sequence_len vocab_size"]:
        logits = self.W_U(x)
        return logits

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: BaseTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.c_proj = nn.Linear(cfg.num_heads * cfg.d_head, cfg.d_model)
        self.c_proj.NANOGPT_SCALE_INIT = 1 #type: ignore
        self.n_head = cfg.num_heads
        self.d_head = cfg.d_head

    #@jaxtyped(typechecker=beartype) TODO torch.compile doesn't work with jaxtyped decorators 
    def forward(
        self, 
        x: Float[Tensor, "batch sequence_len d_model"],
        attn_mask: Optional[Bool[Tensor, "batch sequence_len sequence_len"]] = None,
    ) -> Float[Tensor, "batch sequence_len d_model"]:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        
        attn = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None if self.cfg.is_causal else attn_mask, 
            is_causal=self.cfg.is_causal
        )
        y = attn.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, cfg: BaseTransformerConfig):
        super().__init__()
        hidden_dim = cfg.d_model*cfg.d_mult

        self.w1 = nn.Linear(cfg.d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, cfg.d_model, bias=False)
        self.w3 = nn.Linear(cfg.d_model, hidden_dim, bias=False)
        self.act_fn = cfg.nonlin

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        """
        RMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.variance_epsilon = eps

    def forward(self, x : Float[Tensor, "batch sequence_len d_model"]):
        input_dtype = x.dtype

        #we perform normalization in float32
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)
    
from typing import Protocol

class SwiGlUConfig(Protocol):
    d_model: int
    nonlin: Union[nn.Module, Callable[[Tensor], Tensor]] #NOTE llama3 uses SILU 
    multiple_of: int
    ffn_dim_multiplier: Optional[float]

#LLAMA3 ffn layer
class SwiGLU_MLP(nn.Module):
    def __init__(
        self,
        config : SwiGlUConfig
    ):
        super().__init__()
        self.nonlin = config.nonlin
        hidden_dim = int(2 * config.d_model / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        self.w1 = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, hidden_dim, bias=False)

    def forward(
        self, 
        x : Float[Tensor, "batch sequence_len d_model"]
    ) -> Float[Tensor, "batch sequence_len d_model"]:
        return self.w2(self.nonlin(self.w1(x)) * self.w3(x))


#roatry embeddings from huggingface
class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        inv_freq = inv_freq
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, x, cos, sin):
        cos = cos[:, :, : x.shape[-2], :]
        sin = sin[:, :, : x.shape[-2], :]

        return (x * cos) + (self.rotate_half(x) * sin)

    def _update_cos_sin_tables(self, x, seq_dimension=2):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)

        return (
            self.apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            self.apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )

#from https://github.com/meta-llama/llama3/blob/bf8d18cd087a4a0b3f61075b7de0b86cf6c70697/llama/model.py#L90
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

ACT2CLS = {
    "gelu": (nn.GELU, {'approximate': 'tanh'}) ,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
}

class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)

ACT2FN = ClassInstantier(ACT2CLS)




class GatedMLP(nn.Module):
    def __init__(
        self,
        in_features : int,
        hidden_features : int,
        out_features : Optional[int] = None,
        activation=F.silu,
        bias=False,
        multiple_of=128,
    ):
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else int(8 * in_features / 3)
        )
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)
        y = self.fc2(y)
        return y
    

T = TypeVar('T', bound='ModelMixin')

class ModelOutputMixin(BaseModel):
    logits : Tensor
    loss : Optional[Tensor] = None

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def check_tensor_values(self) -> 'Self':
        for field_name, value in self.model_dump().items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any() or torch.isinf(value).any():
                    raise ValueError(f"Tensor '{field_name}' contains NaN or Inf values.")
        return self


class ModelMixin(nn.Module, ABC):
    def __init__(
        self, 
        cfg : ModelConfig,
        is_master_process : bool
    ):
        super().__init__()
        assert isinstance(is_master_process, bool)
        self.is_master_process = is_master_process
        self.cfg = cfg

    def to(self, *args, **kwargs):
        device = kwargs.get('device')
        if device is not None:
            self.device = device
        return super().to(*args, **kwargs)
    
    def configure_optimizers(
        self, 
        weight_decay : float, 
        lr_config : LRConfig, 
        device_type : str,
        betas : tuple[float, float] = (0.9, 0.95)
    ):
        
        print("Non-embedding parameter count:", count_non_embedding_params(self))
        total_count = sum(p.numel() for p in self.parameters())
        print("Number of embedding params: ", total_count - count_non_embedding_params(self))
        print("Total parameter count:", total_count)
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight Tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if self.is_master_process:
            print(f"num decayed parameter Tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter Tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if self.is_master_process:
            print(f"using fused AdamW: {use_fused}")

        optim_kwargs = {
            "lr": lr_config.max_lr,
            "betas": betas,
            "eps": 1e-8
        }
        if self.cfg.use_adam_8_bit:
            optimizer = bnb.optim.AdamW8bit(
                optim_groups, 
                **optim_kwargs
            )
        else:
            if lr_config.schedule_free:
                optimizer = AdamWScheduleFree(
                    optim_groups, 
                    **optim_kwargs
                )
            else:
                optimizer = torch.optim.AdamW(
                    optim_groups, 
                    **optim_kwargs,
                    fused=use_fused
                )
        return optimizer
    
    def save_model_with_metadata(
        self, 
        path: str, 
        optimizer: torch.optim.Optimizer, 
        lr : float,
        step: int
    ):
        assert self.cfg is not None, "model has to have attribute config"

        torch.save({
            'model': self.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': self.cfg if not isinstance(self.cfg, dict) else self.cfg.model_dump(),
            'config_type': self.cfg.__class__.__name__,
            'lr': lr,
            'step': step,
            'rng_state': torch.get_rng_state()
        }, path)


    @classmethod
    @overload
    def load_pretrained(
        cls: Type[T],
        model_path: str,
        is_master_process: bool,
        lr_config: LRConfig,
        with_metadata: Literal[False] = False,
    ) -> T: ...

    @classmethod
    @overload
    def load_pretrained(
        cls: Type[T],
        model_path: str,
        is_master_process: bool,
        lr_config: LRConfig,
        with_metadata: Literal[True] = True,
    ) -> tuple[
            T, 
            torch.optim.Optimizer, 
            float
        ]: ...

    @classmethod
    def load_pretrained(
        cls : Type[T], 
        model_path: str, 
        is_master_process: bool,
        lr_config : LRConfig,
        with_metadata: bool = False,
        ) -> Union[
            T, 
            tuple[
                T, 
                torch.optim.Optimizer, 
                float
            ]
        ]:
        state_dict = torch.load(model_path, map_location='cpu')
        config = state_dict['config']
        config_type = state_dict['config_type']
        config_cls = CONFIG_MAP[config_type]
        if isinstance(config, dict):
            config = config_cls(**config)
        model = cls(config, is_master_process)
        model.load_state_dict(state_dict['model'])
        
        if not with_metadata:
            return model
        else:
            optimizer = model.configure_optimizers(weight_decay=0.0, lr_config=lr_config, device_type="cpu")
            optimizer.load_state_dict(state_dict['optimizer'])
            rng_state = state_dict['rng_state']
            torch.set_rng_state(rng_state)
            lr = state_dict['lr']
            return (model, optimizer, lr)

    @abstractmethod
    def forward(
        self, 
        idx : Float[Tensor, "B T"], 
        targets : Optional[Int[Tensor, "B T"]] = None
    ) -> ModelOutputMixin:...

    @abstractmethod
    def _init_weights(self):...

class VanillaTransformerOutput(ModelOutputMixin):
    pass

class TransformerMixin(ModelMixin):
    def __init__(
        self, 
        cfg : BaseTransformerConfig,
        is_master_process : bool
    ):
        super().__init__(cfg, is_master_process) 
        """ if cfg.use_adam_8_bit:
            #this is adviced when using 8 bit optimizer
            self.embedding = bnb.nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.pos_embed = bnb.nn.Embedding(cfg.n_ctx, cfg.d_model)
        else:
            self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.pos_embed = nn.Embedding(cfg.n_ctx, cfg.d_model) """

        cfg.model_validate(cfg.model_dump())#this is for testing it is either instance or inherits
        self.cfg = cfg

    def _init_weights(self, module):
        #super()._init_weights(module)  TODO add this
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.cfg.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def check_forward(
        self, 
        idx : Float[Tensor, "B T"], 
        targets : Optional[Int[Tensor, "B T"]] = None
    ):
        T = idx.shape[1]
        assert T <= self.cfg.n_ctx, f"Cannot forward sequence of length {T}, block size is only {self.cfg.n_ctx}"
        if torch.any(idx >= self.cfg.vocab_size):
            raise ValueError(f"Index out of range. Max index should be {self.cfg.vocab_size - 1} got {idx.max()}")
        if targets is not None:
            assert targets.shape[0] == idx.shape[0], f"Targets and input idx have to have the same batch size, got {targets.shape[0]} and {idx.shape[0]}"
            
class VanillaTransformerBlock(nn.Module):
    def __init__(self, cfg : BaseTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.ln = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    #@jaxtyped(typechecker=beartype) TODO torch.compile doesn't work with jaxtyped decorators
    def forward(
        self, 
        x : Float[Tensor, "batch sequence_len d_model"],
    )-> Float[Tensor, "batch sequence_len d_model"]:
        x = x + self.attn(self.ln(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class VanillaTransformer(TransformerMixin):
    def __init__(
        self, 
        cfg : BaseTransformerConfig, 
        is_master_process : bool = True
    ):
        super().__init__(
            is_master_process=is_master_process,
            cfg=cfg
        )
        self.transformer = nn.ModuleDict(dict(
            embedding = nn.Embedding(cfg.vocab_size, cfg.d_model),
            pos_embed = nn.Embedding(cfg.n_ctx, cfg.d_model),
            layers = nn.ModuleList([VanillaTransformerBlock(cfg) for _ in range(cfg.n_layers)]),
            ln_f = nn.LayerNorm(cfg.d_model),
        ))
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.transformer.embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)

    #@jaxtyped(typechecker=beartype) TODO torch.compile doesn't work with jaxtyped decorators
    def forward(
        self, 
        idx : Int[torch.Tensor, "B T D"], 
        targets : Optional[Int[torch.Tensor, "B T D"]] = None
    ) -> VanillaTransformerOutput:
        self.check_forward(idx, targets)
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.pos_embed.forward(pos) # position embeddings of shape (T, d_model)
        tok_emb = self.transformer.embedding.forward(idx) # token embeddings of shape (B, T, d_model)
        print("tok_emb", tok_emb.shape)
        print("pos_emb", pos_emb.shape)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in cast(List[VanillaTransformerBlock], self.transformer.layers):
            x = block.forward(x)
        x = self.transformer.ln_f(x)
        x = self.lm_head(x)
        if targets is None:
            return VanillaTransformerOutput(logits=x)
        else:
            return VanillaTransformerOutput(
                logits=x, 
                loss=F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))
            )
   