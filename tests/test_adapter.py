import os
import sys
# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from dataclasses import dataclass
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import pytest
from .test_models import get_test_cases, get_base_cases, get_hf_cases, hooked_transformer_cfg
from Auto_HookPoint import HookedTransformerAdapter
from typing import TypeVar, Dict
import torch
from unittest.mock import patch, MagicMock
import torch.nn as nn

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
def mock_auto_config():
    with patch('transformers.AutoConfig') as mock_config:
        mock_instance = AutoConfig.from_pretrained("gpt2")
        mock_config.from_pretrained.return_value = mock_instance
        yield mock_config

def test_adapter_init_hf(mock_auto_config):

    @dataclass
    class Cfg:
        device: str = "cpu"
        vocab_size: int = mock_auto_config.vocab_size
        n_ctx: int = 10  # dummy value
    
    try:
        model = HookedTransformerAdapter(
            hf_model_name="gpt2",
            cfg=Cfg(),
            embedding_attr='_module.transformer._module.wte._module'
        )
        assert isinstance(model, HookedTransformerAdapter)
    except Exception as e:
        pytest.fail(f"Error initializing adapter: {e}")

def test_adapter_init_base_case():
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(gpt2_tokenizer.vocab_size, 10)
            self.linear = nn.Linear(10, 10)

        def forward(self, x):
            x = self.emb(x)
            return self.linear(x)
        
    @dataclass
    class Config:
        n_ctx: int = 12
        device: str = "cpu"
        vocab_size: int = 50257

    try:
        model = HookedTransformerAdapter(
            model=MyModule(), 
            tokenizer=gpt2_tokenizer, 
            cfg=Config(),
            embedding_attr='_module.emb._module'
        )   

        model = HookedTransformerAdapter(
            model=MyModule(), 
            tokenizer=gpt2_tokenizer, 
            cfg=Config(),
            embedding_attr=None #NOTE this is the default
        )
    except Exception as e:
        assert False, f"Error initializing adapter: {e}"