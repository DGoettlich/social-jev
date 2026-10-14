"""Offline Qwen3 execution with the trained scalar linear head."""

import torch
from torch import nn
from transformers import PreTrainedModel, Qwen3Model

from .config import SocialJevConfig
from .data import Batch


class LinearModel(PreTrainedModel):
    """Load the backbone and linear weights from a SocialJev checkpoint.

    Args:
        config: Qwen3 configuration for the linear classifier.

    Examples:
        >>> model = LinearModel.from_pretrained(repo)  # doctest: +SKIP
    """

    config_class = SocialJevConfig
    """Configuration class for linear checkpoints."""
    base_model_prefix = "backbone"
    """Checkpoint prefix for backbone parameters."""
    _supports_sdpa = True
    """Enable the backbone's scaled dot-product attention implementation."""

    def __init__(self, config: SocialJevConfig):
        """Construct the Qwen3 backbone and scalar readout."""
        super().__init__(config)
        self.backbone = Qwen3Model(config)
        self.decision_head = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(self, input_ids, attention_mask):
        """Return raw logits from each candidate's last nonpadding token."""
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        positions = attention_mask.sum(-1) - 1
        return self.decision_head(hidden[rows, positions]).squeeze(-1)


class TransformersEngine:
    """Run a collection of candidate sequences as one padded CPU batch.

    Args:
        repo: Self-contained checkpoint path or Hugging Face repository.
        tokenizer: Checkpoint tokenizer used for padding.

    Examples:
        >>> engine = TransformersEngine(repo, tokenizer)  # doctest: +SKIP
        >>> engine.score([])  # doctest: +SKIP
        []
    """

    def __init__(self, repo, tokenizer):
        """Load the checkpoint in its saved dtype on CPU."""
        model, loading = LinearModel.from_pretrained(repo, dtype="auto", output_loading_info=True)
        if loading["missing_keys"] or loading["unexpected_keys"]:
            message = "Checkpoint weights do not match the backbone and trained linear head"
            raise ValueError(message)
        self.model = model.eval()
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id

    @torch.inference_mode()
    def score(self, inputs: list[list[int]]) -> list[float]:
        """Pad candidates and return ordered scalar logits."""
        if not inputs:
            return []
        assert self.model is not None
        batch = Batch.from_inputs(inputs, self.pad_id)
        return (
            self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
            .float()
            .tolist()
        )

    def stop(self):
        """Release the model weights."""
        self.model = None
