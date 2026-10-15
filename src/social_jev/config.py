"""Checkpoint configuration for the supported Qwen3 scalar classifier."""

from dataclasses import asdict

from transformers import AutoConfig, Qwen3Config

from .head import DecisionHeadConfig


class SocialJevConfig(Qwen3Config):
    """Configure a Qwen3 backbone with a trained scalar linear or MLP readout.

    Args:
        decision_head: Saved architecture identifier; describes a linear or MLP head.
        **kwargs: Qwen3 backbone configuration.

    Examples:
        >>> config = SocialJevConfig(hidden_size=16)
        >>> config.decision_head
        {'type': 'linear', 'dims': [1], 'activation': 'gelu', 'layernorm': False}
    """

    model_type = "social_jev"
    """Architecture identifier used by the checkpoint and vLLM plugin."""

    def __init__(self, *, decision_head=None, **kwargs):
        """Initialize the supported backbone and saved scalar architecture."""
        super().__init__(**kwargs)
        self.decision_head = asdict(DecisionHeadConfig(**(decision_head or {})))
        self.tie_word_embeddings = False


AutoConfig.register("social_jev", SocialJevConfig)
