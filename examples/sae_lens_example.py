import torch
from sae_lens import SAETrainingRunner, LanguageModelSAERunnerConfig
from transformers import AutoConfig
from dataclasses import dataclass
import os
import sys
# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Auto_HookPoint import HookedTransformerAdapter, HookedTransformerAdapterCfg


#most of the credit for this example goes to https://gist.github.com/joelburget

## An important thing to NOTE. 
# 1. This will not work for all hook_points as run_with_cache assumes 
#    the input to a hookpoint is always a tensor and many 
#    modules in hf-transformers return tuples or other non-tensor objects.
#    hopefully over time this will be fixed 
# 2. for tensors sae_lens when building 
#    the activationstore(the training data) 
#    assumes the input has shape [batch, sequence_length, hidden_size]
#    so hookpoints like on the gate of the MixtralSparseMoeBlock 
#    will not work as the shape would be [batch*sequence_length, router_logits]

# Setup
model_name = "joelb/Mixtral-8x7B-1l"
config = AutoConfig.from_pretrained(model_name)

total_training_steps = 15_000  # probably we should do more
batch_size = 4096
total_training_tokens = total_training_steps * batch_size

lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5  # 20% of training
l1_warm_up_steps = total_training_steps // 20  # 5% of training


#sae_lens
cfg = LanguageModelSAERunnerConfig(
    model_name=model_name,
    hook_name="model.norm.hook_point",
    hook_layer=0,
    d_in=config.hidden_size,
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
    lr_warm_up_steps=lr_warm_up_steps,  # this can help avoid too many dead features initially.
    lr_decay_steps=lr_decay_steps,  # this will help us avoid overfitting.
    l1_coefficient=5,  # will control how sparse the feature activations are
    l1_warm_up_steps=l1_warm_up_steps,  # this can help avoid too many dead features initially.
    lp_norm=1.0,  # the L1 penalty (and not a Lp for p < 1)
    train_batch_size_tokens=batch_size,
    context_size=4096,  # will control the lenght of the prompts we feed to the model. Larger is better but slower. so for the tutorial we'll use a short one.
    # Activation Store Parameters
    n_batches_in_buffer=64,  # controls how many activations we store / shuffle.
    training_tokens=total_training_tokens,  # 100 million tokens is quite a few, but we want to see good stats. Get a coffee, come back.
    store_batch_size_prompts=16,
    # Resampling protocol
    use_ghost_grads=False,  # we don't use ghost grads anymore.
    feature_sampling_window=1000,  # this controls our reporting of feature sparsity stats
    dead_feature_window=1000,  # would effect resampling or ghost grads if we were using it.
    dead_feature_threshold=1e-4,  # would effect resampling or ghost grads if we were using it.
    # WANDB
    #log_to_wandb=True,  # always use wandb unless you are just testing code.
    #wandb_log_frequency=30,
    #eval_every_n_wandb_logs=20,
    # Misc
    device="mps",
    seed=42,
    n_checkpoints=0,
    checkpoint_path="checkpoints",
    dtype="float16"
)

if __name__ == "__main__":
    device = "cuda" 
    hooked_model = HookedTransformerAdapter(
        HookedTransformerAdapterCfg(
            device=device, 
            n_ctx=512,
            block_attr="model.layers",
            embedding_attr="model.embed_tokens"
        ),
        model_name
    ).to(device)
    
    print("hooked_model.named_parameters()", list(hooked_model.named_parameters()))
    #print("hooked_model.children()", list(hooked_model.children()))
    print("hooked_model.modules()", list(hooked_model.modules()))
    for param in hooked_model.modules():
        print(param.device)
    # Update the cfg to use the same device
    cfg.device = device
    
    # Ensure the model's parameters are on the correct device
    hooked_model = hooked_model.to(device)
    
    sparse_autoencoder = SAETrainingRunner(cfg, override_model=hooked_model).run()
   