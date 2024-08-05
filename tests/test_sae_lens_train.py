from sae_lens import SAETrainingRunner, LanguageModelSAERunnerConfig
from transformer_lens import HookedTransformerConfig
from torch import nn
import pytest
import os
import sys

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from tests.test_models import gpt2_tokenizer
from tests.test_adapter import get_test_cases, TestCase
from Auto_HookPoint.utils import get_device
from Auto_HookPoint.HookedTransformerAdapter import HookedTransformerAdapter, HookedTransformerAdapterCfg


@pytest.mark.parametrize("test_case", get_test_cases())
def test_train(
    test_case : TestCase
):
    model_name, model, map_cfg, hooked_transformer_cfg, hook_name, layer = test_case
    print(f"training sae lens on {model_name} with hook {hook_name} at layer {layer}")
    device = get_device()
    model = HookedTransformerAdapter(
        adapter_cfg=map_cfg,
        model=model,
        tokenizer=gpt2_tokenizer,
        hooked_transformer_cfg=hooked_transformer_cfg
    )
    cfg = LanguageModelSAERunnerConfig(
        model_name=model_name,
        hook_name=hook_name,
        hook_layer=layer,
        d_in=hooked_transformer_cfg.d_model,
        dataset_path="monology/pile-uncopyrighted",
        is_dataset_tokenized=False,
        streaming=True,  
        mse_loss_normalization=None,  # We won't normalize the mse loss,
        expansion_factor=4,
        b_dec_init_method="zeros",  # The geometric median can be used to initialize the decoder weights.
        apply_b_dec_to_input=False,  # We won't apply the decoder weights to the input.
        normalize_sae_decoder=False,
        scale_sparsity_penalty_by_decoder_norm=True,
        decoder_heuristic_init=True,
        init_encoder_as_decoder_transpose=True,
        normalize_activations="expected_average_only_in",
        # Training Parameters
        lr=5e-5,  # lower the better, we'll go fairly high to speed up the tutorial.
        adam_beta1=0.9,  # adam params (default, but once upon a time we experimented with these.)
        adam_beta2=0.999,
        lr_scheduler_name="constant",  # constant learning rate with warmup. Could be better schedules out there.
        lr_warm_up_steps=1,  # this can help avoid too many dead features initially.
        lr_decay_steps=2,  # this will help us avoid overfitting.
        l1_coefficient=5,  # will control how sparse the feature activations are
        l1_warm_up_steps=1,  # this can help avoid too many dead features initially.
        lp_norm=1.0,  # the L1 penalty (and not a Lp for p < 1)
        train_batch_size_tokens=2,
        context_size=hooked_transformer_cfg.n_ctx,  # will control the lenght of the prompts we feed to the model. Larger is better but slower. so for the tutorial we'll use a short one.
        # Activation Store Parameters
        log_to_wandb=False,
        n_batches_in_buffer=64,  # controls how many activations we store / shuffle.
        training_tokens=10,  # 100 million tokens is quite a few, but we want to see good stats. Get a coffee, come back.
        store_batch_size_prompts=16,
        device=device,
        seed=42,
        dtype="float32",
        autocast=True
    )
    model.to(device)
    runner = SAETrainingRunner(cfg, override_model=model)
    result = runner.run()
    
    assert result is not None, "SAETrainingRunner.run() should return a result"
