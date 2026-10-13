"""Hugging Face LM-head loading and offline batched inference."""

from typing import overload

import torch
from torch.nn.attention.flex_attention import create_mask
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .attention import AttentionMask, Positions
from .config import SocialJevConfig
from .data import Batch, Processor
from .types import Request, Result


class SocialJev(PreTrainedModel):
    config_class = SocialJevConfig
    base_model_prefix = "backbone"
    _supports_sdpa = True
    _supports_flex_attn = True

    def __init__(self, config: SocialJevConfig):
        super().__init__(config)
        causal = AutoModelForCausalLM.from_config(
            config.backbone_config,
            attn_implementation=config._attn_implementation or "sdpa",
        )
        self.backbone = causal.base_model
        self.decision_head = causal.get_output_embeddings()
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        loaded = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        assert isinstance(model, cls)
        tokenizer_options = {
            key: kwargs[key]
            for key in (
                "revision",
                "cache_dir",
                "local_files_only",
                "token",
                "force_download",
                "subfolder",
                "trust_remote_code",
            )
            if key in kwargs
        }
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            config=model.config.backbone_config,
            **tokenizer_options,
        )
        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        model.processor = Processor(tokenizer, model.config)
        return loaded

    def forward(self, batch: Batch) -> torch.Tensor:
        answer_length = len(self.processor.answer_ids)
        mask = AttentionMask(batch, answer_length=answer_length).value
        if self.backbone.config._attn_implementation != "flex_attention":
            size, length = batch.input_ids.shape
            mask = create_mask(
                mask.mask_mod, size, 1, length, length, device=str(batch.input_ids.device)
            )
        hidden = self.backbone(
            input_ids=batch.input_ids,
            position_ids=Positions(batch, answer_length=answer_length).value,
            attention_mask={"full_attention": mask},
            use_cache=False,
        ).last_hidden_state
        indices = batch.option_indices
        positions = batch.option_positions[indices.max(dim=-1).values]
        answers = hidden[positions[:, 0], positions[:, 1]]
        codes = torch.tensor(self.processor.code_token_ids, device=hidden.device)
        scores = self.decision_head(answers)[:, codes[: indices.shape[1]]]
        return scores.masked_fill(indices < 0, -torch.inf)

    @overload
    def query(self, requests: Request, *, batch_size: int = 16) -> list[Result]: ...

    @overload
    def query(self, requests: list[Request], *, batch_size: int = 16) -> list[list[Result]]: ...

    @torch.inference_mode()
    def query(
        self, requests: Request | list[Request], *, batch_size: int = 16
    ) -> list[Result] | list[list[Result]]:
        self.eval()
        single = not isinstance(requests, list)
        items = [requests] if single else requests
        chain = self.processor.batches(items, batch_size=batch_size)
        pending = iter(items)
        results: list[list[Result]] = []
        while (batch := chain.next()) is not None:
            rows = iter(self(batch.to(self.device)))
            for _ in batch.roles:
                request = next(pending)
                answers = []
                for query in request.queries:
                    probabilities = next(rows)[: len(query.options)].float().softmax(0).cpu()
                    answers.append(
                        Result(
                            choice=query.options[int(probabilities.argmax())],
                            probabilities=dict(
                                zip(query.options, probabilities.tolist(), strict=True)
                            ),
                        )
                    )
                results.append(answers)
        return results[0] if single else results


SocialJev.register_for_auto_class("AutoModel")
