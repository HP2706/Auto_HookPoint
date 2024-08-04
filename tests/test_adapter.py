from typing import Callable, Literal, Optional, TypeVar, Dict, Union
import torch
from unittest.mock import patch
import torch.nn as nn
from transformer_lens import HookedTransformerConfig
import pytest
from .test_models import gpt2_tokenizer, small_llama_config, small_mixtral_config
import os
import sys
# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

assert os.getenv('HF_TOKEN') is not None, "HF_TOKEN environment variable is not set"


from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, MixtralForCausalLM, GPT2Config, GPT2LMHeadModel
from Auto_HookPoint import HookedTransformerAdapter, HookedTransformerAdapterCfg, HookedTransformerConfig_From_AutoConfig


small_gpt2_config = GPT2Config(
    n_embd=16,
    n_layer=2,
    n_head=4,
    n_inner=None,
    activation_function='gelu_new',
    n_positions=128,
    n_ctx=128,
    vocab_size=50257,
)

def get_hf_cases()-> list[tuple[str, nn.Module, HookedTransformerAdapterCfg, HookedTransformerConfig]]:
    return [
        (
            "openai-community/gpt2",
            GPT2LMHeadModel(config=small_gpt2_config),
            HookedTransformerAdapterCfg(
                block_attr='transformer.h',
                lm_head_attr='lm_head',
                embedding_attr='transformer.wte',
                positional_embedding_attr='transformer.wpe',
                last_layernorm_attr='transformer.ln_f',
                inter_block_fn = lambda x: x[0]
            ),
            HookedTransformerConfig_From_AutoConfig.from_auto_config(
                small_gpt2_config, 
                attn_only=True,
                normalization_type=None,
                positional_embedding_type='standard',
                d_vocab=small_gpt2_config.vocab_size
            ))
        ,(
            "meta-llama/Llama-2-7b-hf",
            LlamaForCausalLM(config=small_llama_config), 
            HookedTransformerAdapterCfg(
                block_attr='model.layers',
                lm_head_attr='lm_head',
                embedding_attr='model.embed_tokens',
                positional_embedding_attr='model.embed_tokens',
                last_layernorm_attr='model.norm',
                inter_block_fn = lambda x: x[0],
                create_kwargs = lambda cfg, residual: {
                    'position_ids': torch.arange(residual.shape[1], device=residual.device).expand(residual.shape[0], -1)
                }
            ), 
            HookedTransformerConfig_From_AutoConfig.from_auto_config(
                small_llama_config, 
                attn_only=True,
                normalization_type=None,
                positional_embedding_type='rotary',
                d_vocab=small_llama_config.vocab_size
            )

        ),
        (
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            MixtralForCausalLM(config=small_mixtral_config), 
            HookedTransformerAdapterCfg(
                block_attr='model.layers',
                lm_head_attr='lm_head',
                embedding_attr='model.embed_tokens',
                positional_embedding_attr='model.embed_tokens',
                last_layernorm_attr='model.norm',
                inter_block_fn = lambda x : x[0],
                create_kwargs = lambda cfg, residual: {
                    'position_ids': torch.arange(residual.shape[1], device=residual.device).expand(residual.shape[0], -1)
                }
            ),
            HookedTransformerConfig_From_AutoConfig.from_auto_config(
                small_mixtral_config, 
                attn_only=True,
                normalization_type=None,
                positional_embedding_type='rotary',
                d_vocab=small_mixtral_config.vocab_size
            )
        )
    ]

@pytest.mark.parametrize("model_name, model, map_cfg, hooked_transformer_cfg", get_hf_cases())
def test_forward(
    model_name: str, 
    model : nn.Module, 
    map_cfg : HookedTransformerAdapterCfg,
    hooked_transformer_cfg : HookedTransformerConfig
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    adapter_model = HookedTransformerAdapter(
        map_cfg=map_cfg,
        model=model,
        tokenizer=tokenizer,
        hooked_transformer_cfg=hooked_transformer_cfg
    )

    adapter_model.forward(
        torch.randint(1, hooked_transformer_cfg.d_vocab, (1, hooked_transformer_cfg.n_ctx)), 
        attention_mask=torch.tril(torch.ones(hooked_transformer_cfg.n_ctx, hooked_transformer_cfg.n_ctx)),
        return_type='logits'
    )


@pytest.mark.parametrize(
    "model_name, model, map_cfg, hooked_transformer_cfg, hook_name",
    [(*case, hook_name) for case, hook_name in zip(get_hf_cases(), ["model.transformer.h.1.mlp.c_fc.hook_point", "model.layers.0.mlp.hook_point", "model.layers.0.block_sparse_moe.experts.1.w3.hook_point"])]
)       
def test_with_cache(
    model_name: str, 
    model : nn.Module, 
    map_cfg : HookedTransformerAdapterCfg,
    hooked_transformer_cfg : HookedTransformerConfig,
    hook_name : str
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    adapter_model = HookedTransformerAdapter(
        map_cfg=map_cfg,
        model=model,
        tokenizer=tokenizer,
        hooked_transformer_cfg=hooked_transformer_cfg
    )
    print("adapter_model.hook_dict", adapter_model.hook_dict)
    if hook_name not in adapter_model.hook_dict.keys():
        raise ValueError(f"Hook {hook_name} not found in model {model_name}")
    adapter_model.run_with_cache(
        torch.randint(1, hooked_transformer_cfg.d_vocab, (1, hooked_transformer_cfg.n_ctx)), 
        names_filter=hook_name,
        attention_mask=torch.tril(torch.ones(hooked_transformer_cfg.n_ctx, hooked_transformer_cfg.n_ctx)),
        return_type='logits'
    )
