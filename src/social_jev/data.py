"""Pull-based inference preparation, exposed through Processor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from importlib.resources import files
from itertools import islice
from typing import TYPE_CHECKING, Any, cast

import torch

from .types import Request

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from .config import SocialJevConfig


@dataclass
class Sample:
    messages: list[dict[str, Any]]


@dataclass
class TokenizedSample:
    ids: list[int]
    message_boundaries: list[int]
    roles: list[str]


@dataclass
class Batch:
    input_ids: torch.Tensor
    message_boundaries: list[list[int]]
    roles: list[list[str]]

    def to(self, device: torch.device | str) -> Batch:
        return replace(self, input_ids=self.input_ids.to(device))

    @property
    def option_positions(self) -> torch.Tensor:
        return torch.tensor(
            [
                (row, end)
                for row, (boundaries, roles) in enumerate(
                    zip(self.message_boundaries, self.roles, strict=True)
                )
                for end, role in zip(boundaries, roles, strict=True)
                if role == "option"
            ],
            dtype=torch.long,
            device=self.input_ids.device,
        )

    @property
    def option_indices(self) -> torch.Tensor:
        groups: list[list[int]] = []
        index = 0
        for roles in self.roles:
            for role in roles:
                if role == "query":
                    groups.append([])
                elif role == "option":
                    groups[-1].append(index)
                    index += 1
        width = max(map(len, groups))
        return torch.tensor(
            [group + [-1] * (width - len(group)) for group in groups],
            dtype=torch.long,
            device=self.input_ids.device,
        )


class Step[T](ABC):
    @abstractmethod
    def next(self) -> T | None: ...


class RequestStep(Step[Request]):
    def __init__(self, requests: list[Request]):
        self.requests = iter(requests)

    def next(self) -> Request | None:
        return next(self.requests, None)


class FormatStep(Step[Sample]):
    def __init__(self, parent: Step[Request]):
        self.parent = parent

    def next(self) -> Sample | None:
        request = self.parent.next()
        if request is None:
            return None
        messages: list[dict[str, Any]] = [{"role": "state", "content": request.state}]
        for query in request.queries:
            messages.append({"role": "query", "content": query.text})
            messages.extend(
                {"role": "example", "content": example.model_dump()} for example in query.examples
            )
            messages.extend({"role": "option", "content": option} for option in query.options)
        return Sample(messages)


class TokenizeStep(Step[TokenizedSample]):
    def __init__(self, parent: Step[Sample], processor: Processor):
        self.parent = parent
        self.processor = processor

    def next(self) -> TokenizedSample | None:
        sample = self.parent.next()
        if sample is None:
            return None
        return self.tokenize(sample)

    def tokenize(self, sample: Sample) -> TokenizedSample:
        processor = self.processor
        config = processor.config
        ids = cast(
            "list[int]",
            processor.tokenizer.apply_chat_template(
                sample.messages,
                role_start=config.role_start,
                role_end=config.role_end,
                option_codes=config.option_codes,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=False,
            ),
        )
        roles = [message["role"] for message in sample.messages]
        closings = iter(i for i, token in enumerate(ids) if token == processor.end_id)
        boundaries = []
        for index, role in enumerate(roles):
            if role == "example":
                next(closings)
                next(closings)
            boundary = next(closings)
            if role == "option" and (index + 1 == len(roles) or roles[index + 1] != "option"):
                end = boundary + 1 + len(processor.answer_ids)
                if ids[boundary + 1 : end] != processor.answer_ids:
                    message = "Tokenizer template must append the LM-head answer prefix"
                    raise ValueError(message)
                boundary = end - 1
            boundaries.append(boundary)
        if len(ids) > config.backbone_config.max_position_embeddings:
            message = "Request exceeds model context capacity"
            raise ValueError(message)
        return TokenizedSample(ids, boundaries, roles)


class PadStep(Step[list[TokenizedSample]]):
    def __init__(self, parent: Step[TokenizedSample], batch_size: int, pad_id: int):
        if batch_size < 1:
            message = "batch_size must be positive"
            raise ValueError(message)
        self.parent = parent
        self.batch_size = batch_size
        self.pad_id = pad_id

    def next(self) -> list[TokenizedSample] | None:
        samples = list(islice(iter(self.parent.next, None), self.batch_size))
        if not samples:
            return None
        length = max(len(sample.ids) for sample in samples)
        return [
            replace(sample, ids=sample.ids + [self.pad_id] * (length - len(sample.ids)))
            for sample in samples
        ]


class CollateStep(Step[Batch]):
    def __init__(self, parent: Step[list[TokenizedSample]]):
        self.parent = parent

    def next(self) -> Batch | None:
        samples = self.parent.next()
        if samples is None:
            return None
        return Batch(
            torch.tensor([sample.ids for sample in samples], dtype=torch.long),
            [sample.message_boundaries for sample in samples],
            [sample.roles for sample in samples],
        )


class Processor:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, config: SocialJevConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.end_id = cast("int", tokenizer.convert_tokens_to_ids(config.role_end))
        self.answer_ids = tokenizer.encode(config.role_start + "answer\n", add_special_tokens=False)
        self.code_token_ids: list[int] = []
        for code in config.option_codes:
            encoded = tokenizer.encode(
                config.role_start + "answer\n" + code, add_special_tokens=False
            )
            if encoded[:-1] != self.answer_ids or encoded[-1] == tokenizer.unk_token_id:
                message = "Each option code must be one known answer token"
                raise ValueError(message)
            self.code_token_ids.append(encoded[-1])
        if len(set(self.code_token_ids)) != len(self.code_token_ids):
            message = "Option codes must have distinct token IDs"
            raise ValueError(message)

    @staticmethod
    def template() -> str:
        return files("social_jev").joinpath("templates/messages.jinja").read_text()

    def batches(self, requests: list[Request], *, batch_size: int = 16) -> CollateStep:
        if any(
            len(query.options) > len(self.code_token_ids)
            for request in requests
            for query in request.queries
        ):
            message = "Query has more options than available option codes"
            raise ValueError(message)
        source = RequestStep(requests)
        formatted = FormatStep(source)
        tokenized = TokenizeStep(formatted, self)
        padded = PadStep(tokenized, batch_size, self.end_id)
        return CollateStep(padded)

    def encode(self, requests: Request | list[Request]) -> Batch:
        items = [requests] if isinstance(requests, Request) else requests
        if not items:
            message = "encode requires at least one request"
            raise ValueError(message)
        batch = self.batches(items, batch_size=len(items)).next()
        assert batch is not None
        return batch
