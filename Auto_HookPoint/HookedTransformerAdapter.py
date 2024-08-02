import logging
from transformer_lens.hook_points import HookedRootModule,NamesFilter, DeviceType
from transformer_lens.HookedTransformer import HookedTransformerKeyValueCache
from transformer_lens.HookedTransformer import Output, USE_DEFAULT_VALUE
from transformer_lens import ActivationCache
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
from Auto_HookPoint.hook import auto_hook
from typing import Any, Iterable, Protocol, Sequence, Tuple, Union, List, Optional, Literal, cast, overload, Callable
from collections.abc import Sized
from jaxtyping import Float, Int
import torch.nn as nn
from contextlib import contextmanager
from torch.nn import functional as F
import torch
from dataclasses import dataclass

@dataclass
class HookedTransformerAdapterCfg:
    block_attr: Optional[str]
    embedding_attr: Optional[str]
    vocab_size: int = 50257
    n_ctx: int = 12
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.cross_entropy
    preproc_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x
    last_layernorm_attr: Optional[str] = None
    unembed_attr: Optional[str] = None
    return_type: Optional[Literal["logits", "loss", "both"]] = "logits"
    normalization_type: Optional[str] = 'LN'
    device: Optional[str] = None
    output_logits_soft_cap: float = 0.0



class HookedTransformerAdapter(HookedRootModule):
    @overload
    def __init__(
        self,
        cfg: HookedTransformerAdapterCfg,
        hf_model_name: str,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        cfg: HookedTransformerAdapterCfg,
    ) -> None: ...

    def __init__(
        self,
        cfg: HookedTransformerAdapterCfg,
        hf_model_name: Optional[str] = None,
        *,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Union[AutoTokenizer, PreTrainedTokenizer]] = None,
    ):
        """
        Initialize a HookedTransformerAdapter.

        This class adapts a HuggingFace transformer model to work with the
        transformer_lens library, allowing the use of hooks and other
        transformer_lens features like run_with_cache.
        NOTE: this is not yet fully compatible with transformer_lens.HookedTransformer 

        Args:
            hf_model_name (Optional[str]): 
                The name of the Hugging Face model to load.
                If provided, the model and tokenizer will be loaded automatically.
            cfg (Cfg): 
                Configuration object to communicate knowledge ie attributes and 
                function from the input model to the adapter.
            model (Optional[nn.Module]): 
                A pre-loaded Hugging Face model.
                Required if hf_model_name is not provided.
            tokenizer (Optional[Union[AutoTokenizer, PreTrainedTokenizer]]): 
                A pre-loaded tokenizer.
                Required if hf_model_name is not provided.

        Raises:
            ValueError: If the input arguments are invalid or if no embedding matrix is found.

        Note:
            Either provide hf_model_name or both model and tokenizer.
            The cfg parameter is required in both cases.
        """
        super().__init__()
        self.validate_args(hf_model_name, model, tokenizer)

        self.return_type = cfg.return_type
        if isinstance(hf_model_name, str):
            self.model = auto_hook(AutoModelForCausalLM.from_pretrained(hf_model_name))
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        elif isinstance(model, nn.Module) and tokenizer is not None:
            self.model = auto_hook(model)
            self.tokenizer = tokenizer
        else:
            raise ValueError("Invalid input. Provide either a model name (str) or both model and tokenizer objects.")

        self.cfg = cfg
        self.preproc_fn = cfg.preproc_fn
        self.tokenizer.pad_token = self.tokenizer.eos_token #type: ignore
        self.n_ctx = cfg.n_ctx  
        self.loss_fn = cfg.loss_fn
        self.device = cfg.device
        
        #NOTE this is not ideal
        if cfg.unembed_attr is not None:
            self.lm_head = self._get_attr_recursively(self.model, cfg.unembed_attr)
        if cfg.last_layernorm_attr is not None:
            self.ln_f = self._get_attr_recursively(self.model, cfg.last_layernorm_attr)
        if cfg.embedding_attr is not None:
            self.W_E = self._get_attr_recursively(self.model, cfg.embedding_attr)
            setattr(self.W_E, 'device', cfg.device) #this is for sae_lens to work
        self.setup()

    def validate_args(
        self, 
        hf_model_name: Optional[str], 
        model: Optional[nn.Module], 
        tokenizer: Optional[Union[AutoTokenizer, PreTrainedTokenizer]]
    ):
        if (hf_model_name is not None) == (model is not None or tokenizer is not None):
            raise ValueError("Provide either a model name or both model and tokenizer objects, not both or neither.")

    def _get_model_blocks(self) -> Sequence[nn.Module]:
        var_name = self.cfg.block_attr
        if var_name is None:
            raise ValueError("block_attr is required when using start_at_layer and stop_at_layer")
        else:
            out = self._get_attr_recursively(self.model, var_name)
            assert isinstance(out, Iterable), f"Expected an iterable of modules, got {type(out)}"
            assert isinstance(out, Sized), f"Expected module to be have attribute __len__, got {type(out)}"
            return cast(Sequence[nn.Module], out)     

    from transformer_lens import HookedTransformer
    def forward(
        self, 
        input,
        return_type: Literal["logits"] = "logits",
        loss_per_token: bool = False,
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
        start_at_layer: Optional[int] = None,
        tokens: Optional[Int[torch.Tensor, "batch pos"]] = None,
        shortformer_pos_embed: Optional[Float[torch.Tensor, "batch pos d_model"]] = None,
        attention_mask: Optional[torch.Tensor] = None,  # [batch pos]
        stop_at_layer: Optional[int] = None,
        past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    ):
        blocks = self._get_model_blocks()
        # Handle pre-block operations (e.g., embedding)
        residual = self.preproc_fn(input)

        if start_at_layer is None:
            start_at_layer = 0

        blocks_and_idxs = list(zip(range(len(blocks)), blocks))
        for i, block in blocks_and_idxs[start_at_layer:stop_at_layer]:
            residual = block(residual)

        if stop_at_layer is not None:
            return residual
        
        if self.cfg.normalization_type is not None:
            residual = self.ln_f(residual)  # [batch, pos, d_model]
        
        if self.return_type is None:
            return None
        else:
            assert self.lm_head is not None
            logits = self.lm_head(residual)  # [batch, pos, d_vocab]
            if self.cfg.output_logits_soft_cap > 0.0:
                logits = self.cfg.output_logits_soft_cap * F.tanh(
                    logits / self.cfg.output_logits_soft_cap
                )
            if self.return_type == "logits":
                return logits
            else:
                assert (
                    tokens is not None
                ), "tokens must be passed in if return_type is 'loss' or 'both'"
                loss = self.loss_fn(logits, tokens)
                if self.return_type == "loss":
                    return loss
                elif self.return_type == "both":
                    return Output(logits, loss)
                else:
                    logging.warning(f"Invalid return_type passed in: {self.return_type}")
                    return None

    def _get_attr_recursively(self, obj, attr : str) -> Any:
        attrs = attr.split('.')
        for a in attrs:
            obj = getattr(obj, a)
        return obj

    def setup(self):
        '''
        Override HookedRootModule.setup 
        to avoid the _module wrapper in names
        '''
        self.hook_dict = self.model.hook_dict
        self.mod_dict = self.model.mod_dict
        for name, hook_point in self.hook_dict.items():
            hook_point.name = name

    def to_tokens(
        self,
        input: Union[str, List[str]],
        prepend_bos: Optional[Union[bool, None]] = True,
        padding_side: Optional[Union[Literal["left", "right"], None]] = None,
        move_to_device: bool = True,
        truncate: bool = True,
    ):
        return self.tokenizer(
            input,
            return_tensors="pt",
            padding=True,
            truncation=truncate,
            max_length=self.n_ctx if truncate else None,
        )["input_ids"] #type: ignore
        
    def run_with_cache(
        self, 
        *model_args, 
        names_filter: NamesFilter = None,
        remove_batch_dim: bool = False,
        return_cache_object: Literal[True] = True,
        stop_at_layer: Optional[int] = None,
        start_at_layer: Optional[int] = None,
        **model_kwargs
    ):

        out, cache_dict = super().run_with_cache(
            *model_args, 
            remove_batch_dim=remove_batch_dim,
            names_filter=names_filter,
            start_at_layer=start_at_layer,
            stop_at_layer=stop_at_layer,
            **model_kwargs
        )

        if return_cache_object:
            cache_dict = ActivationCache(cache_dict, self, has_batch_dim=not remove_batch_dim)
            return out, cache_dict
        else:
            return out, cache_dict