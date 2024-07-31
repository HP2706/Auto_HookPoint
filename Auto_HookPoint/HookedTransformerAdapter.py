from transformer_lens.hook_points import HookedRootModule
from transformer_lens import ActivationCache
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
from Auto_HookPoint.hook import auto_hook
from typing import Protocol, Union, List, Optional, Literal, overload
import torch.nn as nn

class Cfg(Protocol):
    n_ctx: int
    device: str

class HookedTransformerAdapter(HookedRootModule):
    @overload
    def __init__(
        self,
        hf_model_name: str,
        cfg: Cfg,
        *,
        embedding_attr: Optional[str] = None
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        cfg: Cfg,
        embedding_attr: Optional[str] = None
    ) -> None: ...

    def __init__(
        self,
        hf_model_name: Optional[str] = None,
        cfg: Optional[Cfg] = None,
        *,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Union[AutoTokenizer, PreTrainedTokenizer]] = None,
        embedding_attr: Optional[str] = None
    ):
        super().__init__()
        self.validate_args(hf_model_name, model, tokenizer)
        
        if isinstance(hf_model_name, str):
            self.model = auto_hook(AutoModelForCausalLM.from_pretrained(hf_model_name))
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        elif isinstance(model, nn.Module) and tokenizer is not None:
            self.model = auto_hook(model)
            self.tokenizer = tokenizer
        else:
            raise ValueError("Invalid input. Provide either a model name (str) or both model and tokenizer objects.")

        self.cfg = cfg
        self.tokenizer.pad_token = self.tokenizer.eos_token #type: ignore
        self.n_ctx = cfg.n_ctx

        self.W_E = self._find_embedding_matrix(embedding_attr)
        self.setup()

    def validate_args(self, hf_model_name: Optional[str], model: Optional[nn.Module], tokenizer: Optional[Union[AutoTokenizer, PreTrainedTokenizer]]):
        if (hf_model_name is not None) == (model is not None or tokenizer is not None):
            raise ValueError("Provide either a model name or both model and tokenizer objects, not both or neither.")
        

    def _find_embedding_matrix(self, embedding_attr: Optional[str]) -> nn.Parameter:
        if embedding_attr:
            return self._get_attr_recursively(self.model, embedding_attr)

        embedding_matrices = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding):
                print(f'Found embedding matrix: {name}')
                embedding_matrices.append((name, module.weight))
        
        if len(embedding_matrices) == 0:
            raise ValueError("No embedding matrix found in the model.")
        elif len(embedding_matrices) > 1:
            raise ValueError(f"Multiple embedding matrices found: {[name for name, _ in embedding_matrices]}. Please specify the correct attribute using the 'embedding_attr' parameter.")
        
        return embedding_matrices[0][1] 

    def _get_attr_recursively(self, obj, attr):
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
        
    def run_with_cache(self, *model_args, return_cache_object=True, remove_batch_dim=False, **kwargs):
        names_filter = kwargs.get("names_filter", None)
        out, cache_dict = super().run_with_cache(
            *model_args, remove_batch_dim=remove_batch_dim, names_filter=names_filter
        )
        if return_cache_object:
            cache = ActivationCache(cache_dict, self, has_batch_dim=not remove_batch_dim)
            return out, cache
        else:
            return out, cache_dict

    def forward(self, *args, **kwargs):
        result = self.model.forward(*args, **kwargs)
        return result