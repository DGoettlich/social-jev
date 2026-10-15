"""Offline Qwen3 execution with the trained scalar linear or MLP head."""

import torch
from transformers import PreTrainedModel, Qwen3Model

from .config import SocialJevConfig
from .data import Batch
from .head import DecisionHeadConfig, ScalarHead


class ScalarModel(PreTrainedModel):
    """Load the backbone and scalar-head weights from a SocialJev checkpoint.

    Args:
        config: Qwen3 configuration for the scalar classifier.

    Examples:
        >>> model = ScalarModel.from_pretrained(repo)  # doctest: +SKIP
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
        self.backbone = Qwen3Model(config)
        self.decision_head = ScalarHead(
            DecisionHeadConfig(**config.decision_head), config.hidden_size
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
        tokenizer: Checkpoint tokenizer used for padding.

    Examples:
        >>> engine = TransformersEngine(repo, tokenizer)  # doctest: +SKIP
        >>> engine.score([])  # doctest: +SKIP
        []
    """

    def __init__(self, repo, tokenizer, **kwargs):
        """Load the checkpoint using caller dtype and device-placement settings."""
        model, loading = ScalarModel.from_pretrained(
            repo, output_loading_info=True, **{"dtype": "auto", **kwargs}
        )
        if loading["missing_keys"] or loading["unexpected_keys"]:
            message = "Checkpoint weights do not match the backbone and trained linear or MLP head"
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
        maximum = self.model.config.max_position_embeddings
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
