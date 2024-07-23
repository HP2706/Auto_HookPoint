from Auto_HookPoint import auto_hook, check_auto_hook
from transformers.models.mamba.modeling_mamba import MambaForCausalLM, MambaConfig
from transformer_lens.hook_points import HookedRootModule
from typing import cast
import torch

mamba_cfg = MambaConfig(
    vocab_size=1000,
    hidden_size=64,
    state_size=8,
    num_hidden_layers=4,
    layer_norm_epsilon=1e-5,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
    expand=2,
    conv_kernel=2,
    use_bias=False,
    use_conv_bias=True,
    hidden_act="silu",
    initializer_range=0.1,
    residual_in_fp32=False,
    time_step_scale=1.0,
    time_step_min=0.001,
    time_step_max=0.1,
    time_step_init_scheme="random",
    time_step_floor=1e-4,
    rescale_prenorm_residual=False,
    use_cache=True,
)

model = MambaForCausalLM(mamba_cfg)

model = MambaForCausalLM(mamba_cfg)
model = auto_hook(model)
print('model.mod_dict', model.hook_d_dict.keys())

class Model(MambaForCausalLM, HookedRootModule):
    pass

model = cast(Model, model)

input_kwargs = {'input_ids': torch.randint(0, 10, (10, 10)), 'labels': torch.randint(0, 10, (10, 10)),'return_dict': True}

check_auto_hook(MambaForCausalLM, input_kwargs, init_kwargs={'config': mamba_cfg})
