import logging
from transformer_lens.hook_points import HookedRootModule,NamesFilter, DeviceType
from transformer_lens.HookedTransformer import Output, USE_DEFAULT_VALUE
from transformer_lens import ActivationCache
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
from Auto_HookPoint.hook import auto_hook
from typing import Any, Iterable, Protocol, Sequence, Tuple, Union, List, Optional, Literal, cast, overload, Callable
from collections.abc import Sized
import torch.nn as nn
from contextlib import contextmanager
from torch.nn import functional as F
import torch
from jaxtyping import Int

class Cfg(Protocol):
    n_ctx: int
    device: str
    embedding_attr: Optional[str] = None #attribute name of the embedding matrix
    block_attr: Optional[str] = None  #attribute name of the transformer blocks
    last_layernorm_attr : Optional[str] = None
    unembed_attr : Optional[str] = None
    return_type : Optional[Literal["logits", "loss", "both"]] = None
    normalization_type : Optional[str] = 'LN',
    output_logits_soft_cap : float = 0.0
    preproc_fn : Callable[[Any], Any] = lambda x: x
    loss_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.cross_entropy

class HookedTransformerAdapter(HookedRootModule):
    @overload
    def __init__(
        self,
        cfg: Cfg,
        hf_model_name: str,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        cfg: Cfg,
    ) -> None: ...

    def __init__(
        self,
        cfg: Cfg,
        hf_model_name: Optional[str] = None,
        *,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Union[AutoTokenizer, PreTrainedTokenizer]] = None,
    ):
        """
        Initialize a HookedTransformerAdapter.

        This class adapts a Hugging Face transformer model to work with the
        transformer_lens library, allowing the use of hooks and other
        transformer_lens features like run_with_cache.

        Args:
            hf_model_name (Optional[str]): The name of the Hugging Face model to load.
                If provided, the model and tokenizer will be loaded automatically.
            cfg (Optional[Cfg]): Configuration object.
            model (Optional[nn.Module]): A pre-loaded Hugging Face model.
                Required if hf_model_name is not provided.
            tokenizer (Optional[Union[AutoTokenizer, PreTrainedTokenizer]]): A pre-loaded tokenizer.
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
        
        #NOTE this is not ideal
        if cfg.unembed_attr is not None:
            self.lm_head = self._get_attr_recursively(self.model, cfg.unembed_attr)
        if cfg.last_layernorm_attr is not None:
            self.ln_f = self._get_attr_recursively(self.model, cfg.last_layernorm_attr)
        if cfg.embedding_attr is not None:
            self.W_E = self._get_attr_recursively(self.model, cfg.embedding_attr)
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

    def forward(
        self, 
        *model_args,
        loss_per_token: bool = False,
        start_at_layer: Optional[int] = None,
        tokens: Optional[Int[torch.Tensor, "batch pos"]] = None,
        stop_at_layer: Optional[int] = None,
    ):
        blocks = self._get_model_blocks()
        # Handle pre-block operations (e.g., embedding)
        
        #TODO make this more elegant
        print("preproc_fn", self.preproc_fn)
        print("model_args", model_args)
        residual = self.preproc_fn(*model_args)

        if start_at_layer is None:
            start_at_layer = 0

        print("start_at_layer", start_at_layer, "stop_at_layer", stop_at_layer)
        blocks_and_idxs = list(zip(range(len(blocks)), blocks))
        for i, block in blocks_and_idxs[start_at_layer:stop_at_layer]:
            residual = block(residual)
            print("block", i, residual.shape)

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
            cache = ActivationCache(cache_dict, self, has_batch_dim=not remove_batch_dim)
            return out, cache
        else:
            return out, cache_dict