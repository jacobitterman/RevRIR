
import os
import json
from typing import Union

from transformers.models.clap.configuration_clap import ClapAudioConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

class CarirRtfConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ClapTextModel`]. It is used to instantiate a CLAP
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the CLAP
    [calp-hsat-fused](https://huggingface.co/laion/clap-hsat-fused) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the CLAP model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ClapTextModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"relu"`,
            `"relu"`, `"silu"` and `"relu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`ClapTextModel`].
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        projection_hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the projection layer. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        projection_dim (`int`, *optional*, defaults to 512)
            Dimension of the projection head of the `ClapTextModelWithProjection`.

    Examples:

    """
    model_type = "carir_rtf_model"

    def __init__(
        self,
        in_dim = -1,
        out_dim = -1,
        hidden_size = -1,
        projection_dim = -1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = -1
        self.is_decoder = False
        self.classic_feature_dim = 0
        if "is_decoder" in kwargs:
            self.is_decoder = kwargs["is_decoder"]
        if "classic_feature_dim" in kwargs:
            self.classic_feature_dim = kwargs["classic_feature_dim"]

        if "num_mid_layers" not in kwargs:  # backword compatibility
            kwargs["num_mid_layers"] = 1
        self.num_mid_layers = kwargs["num_mid_layers"]

        if "classic_feature_dim" not in kwargs:  # backword compatibility
            kwargs["classic_feature_dim"] = 0
        self.classic_feature_dim = kwargs["classic_feature_dim"]

        if "use_bn" not in kwargs:  # backword compatibility
            kwargs["use_bn"] = False
        self.use_bn = kwargs["use_bn"]


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the rtf config dict if we are loading from CarirConfig
        if config_dict.get("model_type") == "carir":
            config_dict = config_dict["rtf_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def from_json_path(cls, rtf_config_path):
        return cls(**json.load(open(rtf_config_path, 'r'))['rtf_config'])


CarirAudioConfig = ClapAudioConfig


class CarirConfig(PretrainedConfig):
    r"""
    [`ClapConfig`] is the configuration class to store the configuration of a [`ClapModel`]. It is used to instantiate
    a CLAP model according to the specified arguments, defining the text model and audio model configs. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the CLAP
    [laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`ClapTextConfig`].
        audio_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`ClapAudioConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and audio projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* paramter. Default is used as per the original CLAP implementation.
        projection_hidden_act (`str`, *optional*, defaults to `"relu"`):
            Activation function for the projection layers.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            Factor to scale the initialization of the model weights.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    """

    model_type = "carir"

    def __init__(
        self,
        rtf_config = None,
        audio_config = None,
        logit_scale_init_value=(1 / 0.07),
        projection_dim=512,
        projection_hidden_act="relu",
        initializer_factor=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if rtf_config is None:
            rtf_config = {}
            logger.info("rtf_config is None. Initializing the CarirRtfConfig with default values.")

        if audio_config is None:
            audio_config = {}
            logger.info("audio_config is None. initializing the CarirAudioConfig with default values.")

        self.rtf_config = CarirRtfConfig(**rtf_config)
        self.audio_config = CarirAudioConfig(**audio_config)
        self.rtf_config.projection_dim = projection_dim
        self.audio_config.projection_dim = projection_dim

        self.rtf_config.projection_hidden_act = projection_hidden_act
        self.audio_config.projection_hidden_act = projection_hidden_act

        self.projection_dim = projection_dim
        self.projection_hidden_act = projection_hidden_act
        self.hidden_size = self.rtf_config.hidden_size

        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = initializer_factor
        self.num_hidden_layers = self.rtf_config.num_hidden_layers + len(self.audio_config.depths)

        # backword compatibility
        if "use_orig_loss" not in kwargs:
            kwargs["use_orig_loss"] = True
        self.use_orig_loss = kwargs["use_orig_loss"]

        if "hinge_loss_margin" not in kwargs:
            kwargs["hinge_loss_margin"] = 0.5
        self.hinge_loss_margin = kwargs["hinge_loss_margin"]

        if "dual_loss" not in kwargs:
            kwargs["dual_loss"] = False
        self.dual_loss = kwargs["dual_loss"]


    @classmethod
    def from_rtf_audio_configs(cls, rtf_config: CarirRtfConfig, audio_config, **kwargs):
        r"""
        Instantiate a [`ClapConfig`] (or a derived class) from clap text model configuration and clap audio model
        configuration.

        Returns:
            [`ClapConfig`]: An instance of a configuration object
        """

        return cls(rtf_config=rtf_config.to_dict(), audio_config=audio_config.to_dict(), **kwargs)
