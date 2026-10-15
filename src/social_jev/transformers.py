"""Offline Qwen3 execution with the trained scalar linear or MLP head."""

from typing import TYPE_CHECKING, cast

import torch
from transformers import AutoTokenizer, PreTrainedModel, Qwen3Model

from .config import SocialJevConfig
from .data import Batch
from .head import DecisionHeadConfig, ScalarHead

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class SocialJevModel(PreTrainedModel):
    """Load the backbone and scalar-head weights from a SocialJev checkpoint.

    Args:
        config: Qwen3 configuration for the scalar classifier.

    Examples:
        >>> model = SocialJevModel.from_pretrained(repo)  # doctest: +SKIP
    """

    config_class = SocialJevConfig
    """Configuration class for scalar-head checkpoints."""
    base_model_prefix = "backbone"
    """Checkpoint prefix for backbone parameters."""
    _supports_sdpa = True
    """Enable the backbone's scaled dot-product attention implementation."""

    def __init__(self, config: SocialJevConfig):
        """Construct the Qwen3 backbone and scalar readout."""
        super().__init__(config)
        self.backbone = Qwen3Model(config.backbone_config)
        self.decision_head = ScalarHead(
            DecisionHeadConfig(**config.decision_head), config.backbone_config.hidden_size
        )
        self.post_init()

    def forward(self, input_ids, attention_mask):
        """Return raw logits from each candidate's last nonpadding token."""
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        positions = attention_mask.sum(-1) - 1
        return self.decision_head(hidden[rows, positions])


class TransformersEngine:
    """Run a collection of candidate sequences as one padded batch.

    Args:
        repo: Self-contained checkpoint path or Hugging Face repository.

    Examples:
        >>> engine = TransformersEngine(repo)  # doctest: +SKIP
        >>> engine.score([])  # doctest: +SKIP
        []
    """

    def __init__(self, repo, **kwargs):
        """Load the model and tokenizer from the same checkpoint revision."""
        self.model = SocialJevModel.from_pretrained(repo, **kwargs).eval()
        self.tokenizer = cast(
            "PreTrainedTokenizerBase",
            AutoTokenizer.from_pretrained(repo, revision=kwargs.get("revision")),
        )
        self.pad_id = self.tokenizer.pad_token_id

    @torch.inference_mode()
    def score(self, inputs: list[list[int]]) -> list[float]:
        """Pad candidates and return ordered scalar logits."""
        if not inputs:
            return []
        assert self.model is not None
        maximum = self.model.config.backbone_config.max_position_embeddings
        if any(len(ids) > maximum for ids in inputs):
            message = "Candidate sequence exceeds model context capacity"
            raise ValueError(message)
        batch = Batch.from_inputs(inputs, self.pad_id).to(self.model.device)
        return (
            self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
            .float()
            .tolist()
        )

    def stop(self):
        """Release the model weights."""
        self.model = None
