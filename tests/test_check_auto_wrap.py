import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Auto_HookPoint.check import check_auto_hook
from .test_models import (
    AutoEncoder,
    SimpleModule, 
    SimpleModelWithModuleDict, 
    SimpleNestedModuleList, 
    small_llama_config,
    small_mixtral_config
)
from transformers.models.llama import LlamaForCausalLM
from transformers.models.mixtral import MixtralForCausalLM
import pytest
import torch
from typing import Any, Type, TypeVar, Dict

T = TypeVar('T')

#module instance, input 
def get_test_cases():
    return [
        (SimpleModule, {}, {'x' : torch.randn(1, 10)}),
        (AutoEncoder, {'cfg' : {'d_mlp': 10, 'dict_mult': 1, 'l1_coeff': 1, 'seed': 1}}, {'x' : torch.randn(1, 10)}),
        (SimpleModelWithModuleDict, {}, {'x' : torch.randn(1, 10)}),
        (SimpleNestedModuleList, {}, {'x' : torch.randn(1, 10)}),
        (LlamaForCausalLM, {'config' : small_llama_config}, {'input_ids': torch.randint(0, 10, (10, 10)), 'labels': torch.randint(0, 10, (10, 10)),'return_dict': True}),
        (MixtralForCausalLM, {'config' : small_mixtral_config}, {'input_ids': torch.randint(0, 10, (10, 10)), 'labels': torch.randint(0, 10, (10, 10)),'return_dict': True})
    ]

@pytest.mark.parametrize(
    "module, init_kwargs, input_kwargs", 
    get_test_cases()
)
def test_check_auto_hook(
    module: Type[T], 
    init_kwargs : Dict[str, Any],
    input_kwargs : Dict[str, torch.Tensor]
):
    
    check_auto_hook(module, input_kwargs, init_kwargs, strict=True)