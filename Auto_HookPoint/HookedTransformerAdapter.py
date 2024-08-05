import logging
import warnings
from transformer_lens.hook_points import HookedRootModule,  HookPoint
from transformer_lens.components import PosEmbed, Unembed
from transformer_lens.utilities import devices
from transformer_lens.HookedTransformer import (
    HookedTransformer, 
    HookedTransformerKeyValueCache,
    NON_HF_HOSTED_MODEL_NAMES,
    Output,
    Loss,
    USE_DEFAULT_VALUE
)
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, AutoConfig, PretrainedConfig
from .hook import auto_hook
from typing import Any, Iterable, Protocol, Sequence, Tuple, Union, List, Optional, Literal, cast, overload, Callable
from collections.abc import Sized
from jaxtyping import Float
import torch.nn as nn
import torch
from dataclasses import dataclass
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
import os
from transformer_lens import utils
from jaxtyping import Int
from typing import overload

@dataclass
class HookedTransformerConfig_From_AutoConfig(HookedTransformerConfig):
    '''
    HookedTransformerConfig with a classmethod to create a HookedTransformerConfig 
    from a HuggingFace AutoConfig
    '''
    @classmethod
    def from_auto_config(
        cls, 
        auto_config: PretrainedConfig,
        **kwargs
    ) -> HookedTransformerConfig:
        return HookedTransformerConfig(
            n_layers=auto_config.num_hidden_layers,
            d_model=auto_config.hidden_size,
            n_ctx=auto_config.max_position_embeddings,
            d_vocab=auto_config.vocab_size,
            d_head=auto_config.hidden_size // auto_config.num_attention_heads, #NOTE will this work for all models?
            **kwargs
        )

@dataclass
class HookedTransformerAdapterCfg:
    '''
    Defines the mapping between the input model 
    and the HookedTransformer special attributes
    '''
    mappings: dict[str, Optional[str]] # map parameter attributes
    inter_block_fn: Optional[Callable[[Any], Any]] = lambda x: x[0] 
    # often a block return (hidden_state, ..) we want to call the next block with the hidden_state only
    create_kwargs: Optional[Callable[[HookedTransformerConfig, Any], dict[str, Any]]] = None
    preprocess: Optional[Callable[[Any], torch.Tensor]] = None
    # additional kwargs to pass to the model

class AdaptedPosEmbed(PosEmbed):
    '''
    A class that adapts a regular nn.Embedding PosEmbed to a HookedTransformer PosEmbed
    '''
    @classmethod
    def from_regular_pos_embed(cls, pos_embed: nn.Embedding, cfg : HookedTransformerConfig):
        myclass = cls(cfg)
        myclass.W_pos = pos_embed.weight
        return myclass
    
class AdaptedUnembed(Unembed):
    '''
    A class that adapts a regular nn.Linear Unembed to a HookedTransformer Unembed
    '''
    @classmethod
    def from_regular_unembed(cls, unembed: nn.Linear, cfg: HookedTransformerConfig):
        myclass = cls(cfg)
        # Check if we need to transpose the weight matrix

        myclass.W_U = nn.Parameter(unembed.weight.T)
        
        if unembed.bias is not None:
            myclass.b_U = unembed.bias
        else:
            myclass.b_U = nn.Parameter(torch.zeros((cfg.d_vocab,)))
        return myclass

class HookedTransformerAdapter(HookedTransformer):
    pos_embed : PosEmbed
    embed : nn.Embedding
    unembed : Unembed

    @overload
    def __init__(
        self,
        adapter_cfg: HookedTransformerAdapterCfg,
        hooked_transformer_cfg: HookedTransformerConfig,
        hf_model_name: str,
        *,
        move_to_device: bool = True,
        default_padding_side: Literal["left", "right"] = "right",
    ) -> None: ...

    @overload
    def __init__(
        self,
        adapter_cfg: HookedTransformerAdapterCfg,
        hooked_transformer_cfg: HookedTransformerConfig,
        *,
        model: nn.Module,
        tokenizer: Union[AutoTokenizer, PreTrainedTokenizer],
        move_to_device: bool = True,
        default_padding_side: Literal["left", "right"] = "right",
    ) -> None: ...

    def __init__(
        self,
        adapter_cfg: HookedTransformerAdapterCfg,
        hooked_transformer_cfg: HookedTransformerConfig,
        hf_model_name: Optional[str] = None,
        *,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Union[AutoTokenizer, PreTrainedTokenizer]] = None,
        move_to_device: bool = True,
        default_padding_side: Literal["left", "right"] = "right",
    ):
        """
        Initialize a HookedTransformerAdapter.

        This class adapts a HuggingFace transformer model to work with the
        transformer_lens.HookedTransformer.

        We achieve this by having an init function that assigns the same attributes as 
        transformer_lens.HookedTransformer.HookedTransformer.__init__() 
        the user has to provide a cfg object to map their model attributes to equivalent 
        HookedTransformer attributes.


        Args:
            adapter_cfg (HookedTransformerAdapterCfg): Configuration for adapter setup.
            hooked_transformer_cfg (HookedTransformerConfig): Configuration for HookedTransformer.
            hf_model_name (Optional[str]): Name of the Hugging Face model to load.
            model (Optional[nn.Module]): Pre-loaded Hugging Face model.
            tokenizer (Optional[Union[AutoTokenizer, PreTrainedTokenizer]]): Pre-loaded tokenizer.

        Raises:
            ValueError: If the input arguments are invalid.

        Note:
            Provide either hf_model_name or both model and tokenizer.
            adapter_cfg and hooked_transformer_cfg are always required.
        """
        #only initialize HookedRootModule not HookedTransformer.__init__
        HookedRootModule.__init__(self)
        self.validate_args(hf_model_name, model, tokenizer)

        self.adapter_cfg = adapter_cfg
        self.cfg = HookedTransformerConfig.unwrap(hooked_transformer_cfg)
        if isinstance(hf_model_name, str):
            self.model = auto_hook(AutoModelForCausalLM.from_pretrained(hf_model_name).to(hooked_transformer_cfg.device))
            tokenizer = AutoTokenizer.from_pretrained(
                hf_model_name,
                trust_remote_code=self.cfg.trust_remote_code,
            ) #type: ignore
        elif isinstance(model, nn.Module) and tokenizer is not None:
            self.model = auto_hook(model.to(hooked_transformer_cfg.device))
        else:
            raise ValueError("Invalid input. Provide either a model name (str) or both model and tokenizer objects.")

        self.apply_mappings()
        #this is needed for full transformer_lens compatibility

        if tokenizer is not None:
            self.set_tokenizer(tokenizer, default_padding_side=default_padding_side)
                

        self.tokenizer.pad_token = self.tokenizer.eos_token #type: ignore 
        self.device = hooked_transformer_cfg.device
        
        if self.cfg.use_hook_tokens:
            self.hook_tokens = HookPoint()  # [batch, pos]
      
        self.setup()

        if move_to_device:
            assert self.device is not None, "device is not provided"
            self.to(self.device)

    def setup(self):
        super().setup()

        #we do some renaming of model.model. to model.
        for dict_name in ['hook_dict', 'mod_dict']:
            original_dict = getattr(self, dict_name)
            new_dict = {}
            for k, v in original_dict.items():
                new_key = k.replace("model.model.", "model.")
                if isinstance(v, HookPoint):
                    v.name = new_key
                new_dict[new_key] = v
            setattr(self, dict_name, new_dict)

    def apply_mappings(self):
        for ht_attr, model_attr in self.adapter_cfg.mappings.items():
            if model_attr is None:
                logging.warning(f"No model attribute given for {ht_attr}, this might lead to errors")
                continue
            value = self._get_attr_recursively(self.model, model_attr)
            if ht_attr == 'blocks':
                self.blocks = self.check_blocks(value)
            elif ht_attr == 'unembed':
                self.unembed = AdaptedUnembed.from_regular_unembed(value, self.cfg)
            elif ht_attr == 'embed':
                self.embed = value
                self.hook_embed = HookPoint()
            elif ht_attr == 'pos_embed':
                print("value", value)
                if isinstance(value, nn.Embedding):
                    self.pos_embed = AdaptedPosEmbed.from_regular_pos_embed(value, self.cfg)
                else:
                    print("value", value)
                    self.pos_embed = value
                self.hook_pos_embed = HookPoint()
            elif ht_attr == 'ln_final':
                self.ln_final = value
            else:
                setattr(self, ht_attr, value)

    def forward(
        self,
        input: Union[
            str,
            List[str],
            Int[torch.Tensor, "batch pos"],
            Float[torch.Tensor, "batch pos d_model"],
        ],
        return_type: Optional[str] = "logits",
        loss_per_token: bool = False,
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Literal["left", "right"]] = USE_DEFAULT_VALUE,
        start_at_layer: Optional[int] = None,
        tokens: Optional[Int[torch.Tensor, "batch pos"]] = None,
        shortformer_pos_embed: Optional[Float[torch.Tensor, "batch pos d_model"]] = None, #NOTE NOT USED
        attention_mask: Optional[torch.Tensor] = None,  # [batch pos]
        stop_at_layer: Optional[int] = None,
        past_kv_cache: Optional[HookedTransformerKeyValueCache] = None, #NOTE NOT USED
    ) -> Union[
        None,
        Float[torch.Tensor, "batch pos d_vocab"],
        Loss,
        Tuple[Float[torch.Tensor, "batch pos d_vocab"], Loss],
    ]:
        '''a slightly modified version of HookedTransformer.forward()'''
        with utils.LocallyOverridenDefaults(
            self, prepend_bos=prepend_bos, padding_side=padding_side
        ):
            #NOTE this is in place of input_to_embed()!!
            if start_at_layer is None:
                #THIS IS UGLY
                if self.adapter_cfg.preprocess is not None:
                    tokens, residual = self.adapter_cfg.preprocess(self, input)
                else:
                    #residual = input
                    (
                        residual,
                        tokens,
                        shortformer_pos_embed,
                        attention_mask,
                    ) = self.input_to_embed(
                        input,
                        prepend_bos=prepend_bos,
                        padding_side=padding_side,
                        past_kv_cache=past_kv_cache,
                    )
            else:
                assert type(input) == torch.Tensor
                residual = input
            if start_at_layer is None:
                start_at_layer = 0

            if self.adapter_cfg.create_kwargs is not None:
                kwargs = self.adapter_cfg.create_kwargs(self, residual)
            else: 
                kwargs = {}

            blocks_and_idxs = list(zip(range(self.cfg.n_layers), self.blocks))
            
            assert len(blocks_and_idxs[start_at_layer:stop_at_layer]) > 0, f"start_at_layer and stop_at_layer must be valid got start_at_layer={start_at_layer} and stop_at_layer={stop_at_layer}"
            for i, block in blocks_and_idxs[start_at_layer:stop_at_layer]:  # type: ignore
                residual = block(
                    residual,
                    attention_mask=attention_mask,
                    **kwargs
                )  # [batch, pos, d_model]
                if self.adapter_cfg.inter_block_fn is not None:
                    residual = self.adapter_cfg.inter_block_fn(residual)
                #we use a function to do ensure residual is a Tensor 
                # NOTE in the future this might be done via a hook_point

            if stop_at_layer is not None:
                return residual

            if stop_at_layer is not None:
                # When we stop at an early layer, we end here rather than doing further computation
                return residual
            if self.cfg.normalization_type is not None:
                residual = self.ln_final(residual)  # [batch, pos, d_model]
            if return_type is None:
                return None
            else:
                logits = self.unembed(residual)  # [batch, pos, d_vocab]
                if self.cfg.output_logits_soft_cap > 0.0:
                    logits = self.cfg.output_logits_soft_cap * F.tanh(
                        logits / self.cfg.output_logits_soft_cap
                    )
                if return_type == "logits":
                    return logits
                else:
                    assert (
                        tokens is not None
                    ), "tokens must be passed in if return_type is 'loss' or 'both'"
                    loss = self.loss_fn(logits, tokens, per_token=loss_per_token)
                    if return_type == "loss":
                        return loss
                    elif return_type == "both":
                        return Output(logits, loss)
                    else:
                        logging.warning(f"Invalid return_type passed in: {return_type}")
                        return None
                    
    def run_with_cache(self, *args, **kwargs):
        '''
        A wrapper around HookedTransformer.run_with_cache()
        '''
        names_filter = kwargs.get('names_filter', None)
        if names_filter is None:
            raise ValueError("names_filter cannot be None as not all hooks will work")
        
        if isinstance(names_filter, str) and names_filter not in self.hook_dict.keys():
            #we do not allows names that are not in the hook_dict
            # if names_filter is a function we trust the user to provide a valid function
            raise ValueError(f"names_filter must be a key in hook_dict, got {names_filter}")

        return super().run_with_cache(*args, **kwargs)

    def validate_args(
        self, 
        hf_model_name: Optional[str], 
        model: Optional[nn.Module], 
        tokenizer: Optional[Union[AutoTokenizer, PreTrainedTokenizer]]
    ):
        if (hf_model_name is not None) == (model is not None or tokenizer is not None):
            raise ValueError("Provide either a model name or both model and tokenizer objects, not both or neither.")

    def check_blocks(self, blocks):
        assert isinstance(blocks, Iterable) and isinstance(blocks, Sized), \
            f"Expected an iterable of modules with __len__, got {type(blocks)}"
        assert len(blocks) == self.cfg.n_layers, f"Expected {self.cfg.n_layers} blocks, got {len(blocks)}"
        return list(blocks)

    def _get_attr_recursively(self, obj, attr : str) -> Any:
        attrs = attr.split('.')
        for a in attrs:
            obj = getattr(obj, a)
        return obj

    @property
    def W_E(self) -> Float[torch.Tensor, "d_vocab d_model"]:
        """Convenience to get the embedding matrix."""
        return self.embed.weight

    @property
    def W_K(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        raise NotImplementedError("This method is not implemented for HookedTransformerAdapter")

    @property
    def W_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        raise NotImplementedError("This method is not implemented for HookedTransformerAdapter")

    @property
    def W_V(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        raise NotImplementedError("This method is not implemented for HookedTransformerAdapter")

    @property
    def W_O(self) -> Float[torch.Tensor, "n_layers n_heads d_head d_model"]:
        raise NotImplementedError("This method is not implemented for HookedTransformerAdapter")

    @property
    def W_in(self) -> Float[torch.Tensor, "n_layers d_model d_mlp"]:
        raise NotImplementedError("This method is not implemented for HookedTransformerAdapter")

    @property
    def W_gate(self) -> Union[Float[torch.Tensor, "n_layers d_model d_mlp"], None]:
        raise NotImplementedError("This method is not implemented for HookedTransformerAdapter")

    @property
    def W_out(self) -> Float[torch.Tensor, "n_layers d_mlp d_model"]:
        raise NotImplementedError("This method is not implemented for HookedTransformerAdapter")

    @property
    def b_K(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        raise NotImplementedError("This method is not implemented for HookedTransformerAdapter")

    @property
    def b_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        raise NotImplementedError("This method is not implemented for HookedTransformerAdapter")

    @property
    def b_V(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        raise NotImplementedError("This method is not implemented for HookedTransformerAdapter")

    @property
    def b_O(self) -> Float[torch.Tensor, "n_layers d_model"]:
        raise NotImplementedError("This method is not implemented for HookedTransformerAdapter")

    @property
    def b_in(self) -> Float[torch.Tensor, "n_layers d_mlp"]:
        raise NotImplementedError("This method is not implemented for HookedTransformerAdapter")

    @property
    def b_out(self) -> Float[torch.Tensor, "n_layers d_model"]:
        raise NotImplementedError("This method is not implemented for HookedTransformerAdapter")

    @property
    def QK(self):
        raise NotImplementedError("This method is not implemented for HookedTransformerAdapter")

    @property
    def OV(self):
        raise NotImplementedError("This method is not implemented for HookedTransformerAdapter")
