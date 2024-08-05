from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, MixtralForCausalLM, GPT2Config, GPT2LMHeadModel
from typing import List, NamedTuple
import torch
import torch.nn as nn
from transformer_lens import HookedTransformerConfig
import pytest
from .test_models import small_llama_config, small_mixtral_config
import os
import sys
# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

assert os.getenv('HF_TOKEN') is not None, "HF_TOKEN environment variable is not set"
from Auto_HookPoint import HookedTransformerAdapter, HookedTransformerAdapterCfg, HookedTransformerConfig_From_AutoConfig


small_gpt2_config = GPT2Config(
    n_embd=4,
    n_inner=4,
    n_layer=2,
    n_head=4,
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
                mappings={
                    'blocks': 'transformer.h',
                    'unembed': 'lm_head',
                    'embed': 'transformer.wte',
                    'pos_embed': 'transformer.wpe',
                    'ln_final': 'transformer.ln_f',
                },
                inter_block_fn = lambda x: x[0]
            ),
            HookedTransformerConfig_From_AutoConfig.from_auto_config(
                small_gpt2_config, 
                attn_only=True,
                normalization_type=None,
                positional_embedding_type='standard',
            ))
        ,(
            "meta-llama/Llama-2-7b-hf",
            LlamaForCausalLM(config=small_llama_config), 
            HookedTransformerAdapterCfg(
                mappings={
                    'blocks': 'model.layers',
                    'unembed': 'lm_head',
                    'embed': 'model.embed_tokens',
                    'pos_embed' : 'model.rotary_emb',
                    'ln_final': 'model.norm',
                },
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
            )
        ),
        (
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            MixtralForCausalLM(config=small_mixtral_config), 
            HookedTransformerAdapterCfg(
                mappings={
                    'blocks': 'model.layers',
                    'unembed': 'lm_head',
                    'embed': 'model.embed_tokens',
                    'pos_embed' : None, #DUMMY
                    'ln_final': 'model.norm',
                },
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
            )
        )
    ]

class TestCase(NamedTuple):
    model_name: str
    model: nn.Module
    map_cfg: HookedTransformerAdapterCfg
    hooked_transformer_cfg: HookedTransformerConfig
    hook_name: str
    layer: int

def get_test_cases() -> List[TestCase]:
    base_cases = get_hf_cases()
    hook_names_and_layers = [
        ("model.transformer.h.1.mlp.c_fc.hook_point", 1),
        ("model.layers.0.mlp.hook_point", 0),
        ("model.layers.0.input_layernorm.hook_point", 0)
    ]
    
    return [
        TestCase(*case, hook_name, layer)
        for case, (hook_name, layer) in zip(base_cases, hook_names_and_layers)
    ]

@pytest.mark.parametrize("test_case", get_test_cases())
def test_with_cache(test_case: TestCase):
    model_name, model, map_cfg, hooked_transformer_cfg, hook_name, layer = test_case
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    adapter_model = HookedTransformerAdapter(
        adapter_cfg=map_cfg,
        model=model,
        tokenizer=tokenizer,
        hooked_transformer_cfg=hooked_transformer_cfg
    )
    if hook_name not in adapter_model.hook_dict.keys():
        raise ValueError(f"Hook {hook_name} not found in model {model_name}")
    adapter_model.run_with_cache(
        torch.randint(1, hooked_transformer_cfg.d_vocab, (1, hooked_transformer_cfg.n_ctx)), 
        names_filter=hook_name,
        stop_at_layer=layer,
        attention_mask=torch.tril(torch.ones(hooked_transformer_cfg.n_ctx, hooked_transformer_cfg.n_ctx)),
        return_type='logits'
    )

@pytest.mark.parametrize("test_case", get_test_cases())
def test_forward(test_case: TestCase):
    model_name, model, map_cfg, hooked_transformer_cfg, hook_name, layer = test_case
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    adapter_model = HookedTransformerAdapter(
        adapter_cfg=map_cfg,
        model=model,
        tokenizer=tokenizer,
        hooked_transformer_cfg=hooked_transformer_cfg
    )

    adapter_model.forward(
        torch.randint(1, hooked_transformer_cfg.d_vocab, (1, hooked_transformer_cfg.n_ctx)), 
        attention_mask=torch.tril(torch.ones(hooked_transformer_cfg.n_ctx, hooked_transformer_cfg.n_ctx)),
        stop_at_layer=layer,
        return_type='logits'
    )

#TODO
# test core HookedTransformer functions
# 2. generate
# 3. tokens_to_residual_directions
# 4. tokens_to_logits
# 5. tokens_to_loss
# 6. tokens_to_loss_per_token
# 7. tokens_to_loss_per_token_and_position
# 8. tokens_to_loss_per_position