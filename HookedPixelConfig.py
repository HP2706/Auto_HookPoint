from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import pprint
from pydantic import BaseModel, Field
from transformers.models.vit.modeling_vit import ViTConfig

class HookedPixelCfg(BaseModel):
    """
    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`Union[Tuple[int, int], int]`, *optional*, defaults to (16, 8464)]):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        decoder_num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the decoder.
        decoder_hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the decoder.
        decoder_num_hidden_layers (`int`, *optional*, defaults to 8):
            Number of hidden layers in the decoder.
        decoder_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the decoder.
        mask_ratio (`float`, *optional*, defaults to 0.25):
            The ratio of the number of masked tokens in the input sequence.
        norm_pix_loss (`bool`, *optional*, defaults to `True`):
            Whether or not to train with normalized pixels (see Table 3 in the paper).
    """
    hidden_size : int = 768
    num_hidden_layers : int = 12
    num_attention_heads : int = 12
    intermediate_size : int = 3072
    hidden_act : str = "gelu"
    hidden_dropout_prob : float = 0.1
    attention_probs_dropout_prob : float = 0.1
    initializer_range : float = 0.02
    layer_norm_eps : float = 1e-12
    is_encoder_decoder : bool = False
    image_size : Tuple[int, int] = (16, 8464)
    patch_size : int = 16
    num_channels : int = 3
    qkv_bias : bool = True
    decoder_num_attention_heads : int = 16
    decoder_hidden_size : int = 512
    decoder_num_hidden_layers : int = 8
    decoder_intermediate_size : int = 2048
    mask_ratio : float = 0.25
    norm_pix_loss : bool = True
    pruned_heads : Optional[Dict[int, List[int]]] = None
    output_attentions : bool = False
    output_hidden_states : bool = False
    use_return_dict: bool = True

    #added 
    is_decoder: bool = False

    def to_vit_config(self)->ViTConfig:
        config_dict = self.model_dump()

        config_dict.pop('use_return_dict', None)
        config_dict.pop('is_decoder', None)
        
        return ViTConfig(**config_dict)

    #UTILITY methods taken from:
    #https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/HookedTransformerConfig.py#L24
    @classmethod
    def unwrap(cls, config: Union[Dict, "HookedPixelCfg"]) -> 'HookedPixelCfg':
        """
        Convenience function to avoid duplicate code from a common way config is passed to various components
        """
        return HookedPixelCfg.from_dict(config) if isinstance(config, Dict) else config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HookedPixelCfg':
        """
        Instantiates a `HookedTransformerConfig` from a Python dictionary of
        parameters.
        """
        return cls(**config_dict)

    def to_dict(self):
        return self.__dict__

    def __repr__(self):
        return "HookedTransformerConfig:\n" + pprint.pformat(self.to_dict())
