"""Hugging Face configuration for all SocialJev decision heads."""

from string import ascii_uppercase
from typing import ClassVar

from transformers import AutoConfig, PretrainedConfig


class SocialJevConfig(PretrainedConfig):
    model_type = "social_jev"
    sub_configs: ClassVar = {"backbone_config": AutoConfig}

    def __init__(
        self,
        backbone_config=None,
        decision_head=None,
        option_codes=ascii_uppercase,
        role_start="<|im_start|>",
        role_end="<|im_end|>",
        **kwargs,
    ):
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
