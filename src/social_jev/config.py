"""Checkpoint configuration for the supported Qwen3 linear classifier."""

from transformers import AutoConfig, Qwen3Config


class SocialJevConfig(Qwen3Config):
    """Configure a Qwen3 backbone with one trained scalar linear readout.

    Args:
        decision_head: Saved architecture identifier; must describe a linear head.
        **kwargs: Qwen3 backbone configuration.

    Examples:
        >>> config = SocialJevConfig(hidden_size=16)
        >>> config.decision_head
        {'type': 'linear'}
    """

    model_type = "social_jev"
    """Architecture identifier used by the checkpoint and vLLM plugin."""

    def __init__(self, *, decision_head=None, **kwargs):
        """Initialize the supported backbone and fixed linear architecture."""
        super().__init__(**kwargs)
        self.decision_head = decision_head or {"type": "linear"}
        if self.decision_head != {"type": "linear"}:
            message = "SocialJev requires a linear-head checkpoint"
            raise ValueError(message)
        self.tie_word_embeddings = False


AutoConfig.register("social_jev", SocialJevConfig)
