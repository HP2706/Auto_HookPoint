from einops import rearrange, einsum, repeat
from typing import List, Optional, Tuple, Union
from jaxtyping import Float, Int
from typing_extensions import Literal
from functools import partial
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file

from transformer_lens.HookedTransformer import HookedTransformer, Loss, Output
from transformer_lens.hook_points import HookPoint
from HookedPixelConfig import HookedPixelCfg

class HookedPixel(HookedTransformer):
    def __init__(self,
        cfg: HookedPixelCfg,
        initialize_params: Optional[bool] = False):
        super().__init__(cfg, initialize_params)
    