"""Hugging Face configuration for SocialJev LM-head inference."""

from string import ascii_uppercase
from typing import ClassVar

from transformers import AutoConfig, PretrainedConfig


class SocialJevConfig(PretrainedConfig):
    """Hugging Face configuration for an LM-head SocialJev checkpoint.

    Args:
        backbone_config: Backbone configuration object or serialized configuration.
        decision_head: Saved head configuration; only lm_head is supported.
        option_codes: Single-token answer codes assigned to candidate options.
        role_start: Special token starting a message.
        role_end: Special token ending a message.
        **kwargs: Additional Hugging Face configuration settings.

    Examples:
        >>> config = SocialJevConfig(option_codes="ABC")
        >>> config.decision_head["type"]
        'lm_head'
    """

    model_type = "social_jev"
    """Hugging Face identifier for the SocialJev architecture."""
    sub_configs: ClassVar = {"backbone_config": AutoConfig}
    """Configuration types used by the nested backbone."""

    def __init__(
        self,
        backbone_config=None,
        decision_head=None,
        option_codes=ascii_uppercase,
        role_start="<|im_start|>",
        role_end="<|im_end|>",
        **kwargs,
    ):
        """Initialize the backbone, message delimiters, and answer codes."""
        kwargs.pop("tie_word_embeddings", None)
        super().__init__(**kwargs)
        self.tie_word_embeddings = False
        if backbone_config is None:
            backbone_config = {"model_type": "qwen3"}
        if isinstance(backbone_config, dict):
            values = dict(backbone_config)
            backbone_config = AutoConfig.for_model(values.pop("model_type"), **values)
        self.backbone_config = backbone_config
        self.backbone_config.tie_word_embeddings = False
        if decision_head is not None and decision_head.get("type") != "lm_head":
            message = "Only lm_head checkpoints are supported"
            raise ValueError(message)
        self.decision_head = {"type": "lm_head", "dims": [], "activation": "gelu"}
        self.option_codes = option_codes
        self.role_start = role_start
        self.role_end = role_end


SocialJevConfig.register_for_auto_class()
