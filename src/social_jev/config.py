"""Nested checkpoint configuration shared with mert exports."""

from dataclasses import asdict
from typing import ClassVar

from transformers import AutoConfig, PretrainedConfig, Qwen3Config

from .head import DecisionHeadConfig


class SocialJevConfig(PretrainedConfig):
    """Keep backbone and trained scalar-head settings in separate nested configs.

    Args:
        backbone_config: Qwen3 configuration object or serialized configuration.
        decision_head: Saved linear or MLP architecture settings.
        **kwargs: Hugging Face checkpoint metadata.

    Examples:
        >>> config = SocialJevConfig(backbone_config={"model_type": "qwen3", "hidden_size": 16})
        >>> config.backbone_config.hidden_size
        16
    """

    model_type = "social_jev"
    """Architecture identifier preserved in mert exports."""
    sub_configs: ClassVar = {"backbone_config": AutoConfig}
    """Configuration types used by the nested backbone."""

    def __init__(self, backbone_config=None, decision_head=None, **kwargs):
        """Load the nested backbone and scalar architecture without flattening."""
        if backbone_config is None:
            backbone_config = {"model_type": "qwen3"}
        if isinstance(backbone_config, dict):
            values = dict(backbone_config)
            backbone_config = AutoConfig.for_model(values.pop("model_type"), **values)
        if backbone_config.model_type != "qwen3":
            message = "SocialJev currently supports Qwen3 backbones"
            raise ValueError(message)
        assert isinstance(backbone_config, Qwen3Config)
        self.backbone_config = backbone_config
        super().__init__(**kwargs)
        self.decision_head = asdict(DecisionHeadConfig(**(decision_head or {})))
        self.tie_word_embeddings = False
        self.backbone_config.tie_word_embeddings = False

    def get_text_config(self, decoder=None, encoder=None):
        """Expose the nested text configuration to HF and vLLM introspection."""
        del decoder, encoder
        return self.backbone_config


AutoConfig.register("social_jev", SocialJevConfig)
