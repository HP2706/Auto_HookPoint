from typing import Callable, Literal, Optional, TypeVar, Dict
import torch
from unittest.mock import patch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
import pytest
from .test_models import gpt2_tokenizer
import os
import sys
# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Auto_HookPoint import HookedTransformerAdapter


#TODO test on the same things as auto_hook 

T = TypeVar('T', bound=nn.Module)

@pytest.fixture
def mock_auto_model():
    with patch('transformers.AutoModelForCausalLM') as mock_model:
        mock_instance = AutoModelForCausalLM.from_pretrained("gpt2")
        mock_model.from_pretrained.return_value = mock_instance
        yield mock_model

@pytest.fixture
def mock_auto_tokenizer():
    with patch('transformers.AutoTokenizer') as mock_tokenizer:
        mock_instance = AutoTokenizer.from_pretrained("gpt2")
        mock_tokenizer.from_pretrained.return_value = mock_instance
        yield mock_tokenizer

@pytest.fixture
def config():
    @dataclass
    class Config:
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
        device : Optional[str] = None
        output_logits_soft_cap: float = 0.0

    def create_config(**kwargs):
        return Config(**kwargs)

    return create_config

def test_adapter_init_hf(config):

    cfg = config(embedding_attr='transformer.wte', block_attr='transformer.layers')
    try:
        model = HookedTransformerAdapter(
            cfg=cfg,
            hf_model_name="gpt2",
        )
        assert isinstance(model, HookedTransformerAdapter)
    except Exception as e:
        pytest.fail(f"Error initializing adapter: {e}")

def test_adapter_init_base_case(config):
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(gpt2_tokenizer.vocab_size, 10)
            self.linear = nn.Linear(10, 10)

        def forward(self, x):
            x = self.emb(x)
            return self.linear(x)

    cfg = config(embedding_attr='emb', block_attr=None)

    try:
        model = HookedTransformerAdapter(
            model=MyModule(), 
            tokenizer=gpt2_tokenizer, 
            cfg=cfg
        )   

        model = HookedTransformerAdapter(
            model=MyModule(), 
            tokenizer=gpt2_tokenizer, 
            cfg=cfg
        )
    except Exception as e:
        assert False, f"Error initializing adapter: {e}"


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(gpt2_tokenizer.vocab_size, 10)
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(2)])

    def forward(self, x):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
        return x
    
def test_hooked_transformer_adapter_run_with_cache(config):
    model = MyModule()
    adapted_model = HookedTransformerAdapter(
        model=model,
        tokenizer=gpt2_tokenizer,
        cfg=config(
            device = "cpu",
            preproc_fn=lambda x: model.emb(x), 
            return_type="logits",
            output_logits_soft_cap=0.0,
            normalization_type="expected_average_only_in",
            block_attr="layers",
            embedding_attr="emb",
        )
    )

    input_tensor = torch.randint(0, 50257, (1, 10))
    result = adapted_model.run_with_cache(
        input_tensor, 
        start_at_layer=1,
        stop_at_layer=2
    )
    
    assert result is not None, "run_with_cache should return a result"
    assert isinstance(result, tuple), "run_with_cache should return a tuple"
    assert len(result) == 2, "run_with_cache should return a tuple of length 2"
    assert isinstance(result[0], torch.Tensor), "First element of the result should be a tensor"
    assert isinstance(result[1], dict), "Second element of the result should be a dictionary"
