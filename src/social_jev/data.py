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
    """Formatted request messages before tokenization."""

    messages: list[dict[str, Any]]
    """Role-tagged messages in request order."""

    @classmethod
    def from_request(cls, request: Request) -> Sample:
        """Format a typed request into role-tagged messages."""
        messages: list[dict[str, Any]] = [{"role": "state", "content": request.state}]
        for query in request.queries:
            messages.append({"role": "query", "content": query.text})
            messages.extend(
                {"role": "example", "content": example.model_dump()} for example in query.examples
            )
            messages.extend({"role": "option", "content": option} for option in query.options)
        return cls(messages)


@dataclass
class TokenizedSample:
    """Tokenized request with message boundaries and roles."""

    ids: list[int]
    """Token IDs in message order."""
    message_boundaries: list[int]
    """Inclusive final token index of each message."""
    roles: list[str]
    """Message roles aligned with message boundaries."""

    @classmethod
    def from_sample(cls, sample: Sample, processor: Processor) -> TokenizedSample:
        """Tokenize formatted messages and record their boundaries and roles."""
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
        return cls(ids, boundaries, roles)

    def pad(self, length: int, pad_id: int) -> TokenizedSample:
        """Return a copy padded to the requested token length."""
        return replace(self, ids=self.ids + [pad_id] * (length - len(self.ids)))


@dataclass
class Batch:
    """Padded CPU token tensors and aligned message metadata."""

    input_ids: torch.Tensor
    """Padded token IDs with shape (documents, tokens)."""
    message_boundaries: list[list[int]]
    """Inclusive message end indices for each document."""
    roles: list[list[str]]
    """Message roles for each document, aligned with its boundaries."""

    @classmethod
    def from_samples(cls, samples: list[TokenizedSample]) -> Batch:
        """Collate padded samples into a CPU batch."""
        return cls(
            torch.tensor([sample.ids for sample in samples], dtype=torch.long),
            [sample.message_boundaries for sample in samples],
            [sample.roles for sample in samples],
        )

    def to(self, device: torch.device | str) -> Batch:
        """Return a copy with token tensors transferred to the requested device."""
        return replace(self, input_ids=self.input_ids.to(device))

    @property
    def option_positions(self) -> torch.Tensor:
        """Return the batch row and final token index of every option message."""
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
        """Group option indices by query, using -1 for unavailable candidates."""
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
    """Define the pull-based interface shared by inference pipeline stages.

    Examples:
        >>> source = RequestStep([])
        >>> isinstance(source, Step)
        True
        >>> source.next() is None
        True
    """

    @abstractmethod
    def next(self) -> T | None:
        """Pull the next item, or return None when the upstream source is exhausted."""
        ...


class RequestStep(Step[Request]):
    """Read typed requests in their original order.

    Args:
        requests: Documents and queries to pass through the pipeline.

    Examples:
        >>> source = RequestStep([])
        >>> source.next() is None
        True
    """

    def __init__(self, requests: list[Request]):
        """Initialize the ordered request source."""
        self.requests = iter(requests)

    def next(self) -> Request | None:
        """Pull the next item, or return None when the upstream source is exhausted."""
        return next(self.requests, None)


class FormatStep(Step[Sample]):
    """Orchestrate conversion of requests into formatted samples.

    Args:
        parent: Source of typed requests.

    Examples:
        >>> formatted = FormatStep(RequestStep([]))
        >>> formatted.next() is None
        True
    """

    def __init__(self, parent: Step[Request]):
        """Attach the request source to the formatting stage."""
        self.parent = parent

    def next(self) -> Sample | None:
        """Pull the next item, or return None when the upstream source is exhausted."""
        request = self.parent.next()
        if request is None:
            return None
        return Sample.from_request(request)


class TokenizeStep(Step[TokenizedSample]):
    """Orchestrate tokenization of formatted samples.

    Args:
        parent: Source of formatted samples.
        processor: Tokenizer and checkpoint settings used for tokenization.

    Examples:
        >>> formatted = FormatStep(RequestStep([]))
        >>> tokenized = TokenizeStep(formatted, processor)  # doctest: +SKIP
        >>> tokenized.next() is None  # doctest: +SKIP
        True
    """

    def __init__(self, parent: Step[Sample], processor: Processor):
        """Attach the sample source and tokenizer settings."""
        self.parent = parent
        self.processor = processor

    def next(self) -> TokenizedSample | None:
        """Pull the next item, or return None when the upstream source is exhausted."""
        sample = self.parent.next()
        if sample is None:
            return None
        return TokenizedSample.from_sample(sample, self.processor)


class PadStep(Step[list[TokenizedSample]]):
    """Collect requests and orchestrate padding to the group's longest sequence.

    Args:
        parent: Source of tokenized samples.
        batch_size: Maximum number of documents per group.
        pad_id: Token ID used for padding.

    Examples:
        >>> padded = PadStep(tokenized, batch_size=16, pad_id=0)  # doctest: +SKIP
        >>> padded.next() is None  # doctest: +SKIP
        True
    """

    def __init__(self, parent: Step[TokenizedSample], batch_size: int, pad_id: int):
        """Configure the upstream source, batch size, and padding token."""
        if batch_size < 1:
            message = "batch_size must be positive"
            raise ValueError(message)
        self.parent = parent
        self.batch_size = batch_size
        self.pad_id = pad_id

    def next(self) -> list[TokenizedSample] | None:
        """Pull the next item, or return None when the upstream source is exhausted."""
        samples = list(islice(iter(self.parent.next, None), self.batch_size))
        if not samples:
            return None
        length = max(len(sample.ids) for sample in samples)
        return [sample.pad(length, self.pad_id) for sample in samples]


class CollateStep(Step[Batch]):
    """Orchestrate construction of CPU batches from padded sample groups.

    Args:
        parent: Source of padded sample groups.

    Examples:
        >>> batches = CollateStep(padded)  # doctest: +SKIP
        >>> batch = batches.next()  # doctest: +SKIP
    """

    def __init__(self, parent: Step[list[TokenizedSample]]):
        """Attach the padded sample source to the collator."""
        self.parent = parent

    def next(self) -> Batch | None:
        """Pull the next item, or return None when the upstream source is exhausted."""
        samples = self.parent.next()
        if samples is None:
            return None
        return Batch.from_samples(samples)


class Processor:
    """Expose the pull-based request preparation chain.

    Args:
        tokenizer: Tokenizer containing the checkpoint's chat template.
        config: Model configuration defining message roles and option codes.

    Examples:
        >>> from social_jev import Query, Request
        >>> processor = Processor(tokenizer, config)  # doctest: +SKIP
        >>> request = Request(state="The bus network expanded.", queries=[
        ...     Query(text="Policy area?", options=["Transport", "Education"])
        ... ])
        >>> batch = processor.encode(request)  # doctest: +SKIP
        >>> batch.input_ids.shape[0]  # doctest: +SKIP
        1
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, config: SocialJevConfig):
        """Initialize the tokenizer settings and validate single-token answer codes."""
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
        """Read the bundled LM-head chat template from package resources."""
        return files("social_jev").joinpath("templates/messages.jinja").read_text()

    def batches(self, requests: list[Request], *, batch_size: int = 16) -> CollateStep:
        """Assemble a fresh pull-based chain that produces padded CPU batches."""
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
        """Encode all supplied requests into one padded CPU batch."""
        items = [requests] if isinstance(requests, Request) else requests
        if not items:
            message = "encode requires at least one request"
            raise ValueError(message)
        batch = self.batches(items, batch_size=len(items)).next()
        assert batch is not None
        return batch
