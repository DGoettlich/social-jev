"""Dataclass transformations exposed through a pull-based Processor chain."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from importlib.resources import files
from typing import TYPE_CHECKING, cast

import torch

from .types import Request, Result

if TYPE_CHECKING:
    from collections.abc import Iterator

    from transformers import PreTrainedTokenizerBase


@dataclass
class Sample:
    """A complete request represented as role-tagged messages."""

    messages: list[dict[str, str]]
    """Document, questions, and options in their original order."""

    @classmethod
    def from_request(cls, request: Request) -> Sample:
        """Format a complete typed request into messages."""
        messages = [{"role": "state", "content": request.state}]
        for query in request.queries:
            messages.append({"role": "query", "content": query.text})
            messages.extend({"role": "option", "content": option} for option in query.options or [])
        return cls(messages)


@dataclass
class TokenizedSample:
    """A tokenized request with message boundaries and roles."""

    ids: list[int]
    """Token IDs for the complete request."""
    message_boundaries: list[int]
    """Inclusive terminal token index of each message, matching mert."""
    roles: list[str]
    """Message roles aligned with their token boundaries."""

    @classmethod
    def from_sample(cls, sample: Sample, tokenizer: PreTrainedTokenizerBase) -> TokenizedSample:
        """Apply the checkpoint template once to the complete message list."""
        ids = cast(
            "list[int]",
            tokenizer.apply_chat_template(
                sample.messages,
                tokenize=True,
                return_dict=False,
            ),
        )
        end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        boundaries = [index for index, token in enumerate(ids) if token == end_id]
        roles = [message["role"] for message in sample.messages]
        if len(boundaries) != len(roles) or not boundaries or boundaries[-1] != len(ids) - 1:
            message = "Checkpoint template must terminate each message with <|im_end|>"
            raise ValueError(message)
        return cls(ids, boundaries, roles)

    def __len__(self) -> int:
        """Return the number of tokens."""
        return len(self.ids)

    def __getitem__(self, selection: slice) -> TokenizedSample:
        """Slice messages while preserving mert's token offsets and boundaries."""
        start, stop, _ = selection.indices(len(self.roles))
        stop = max(start, stop)
        offsets = [0, *(end + 1 for end in self.message_boundaries)]
        token_start, token_stop = offsets[start], offsets[stop]
        return TokenizedSample(
            ids=self.ids[token_start:token_stop],
            message_boundaries=[end - token_start for end in self.message_boundaries[start:stop]],
            roles=self.roles[start:stop],
        )

    def __add__(self, other: TokenizedSample) -> TokenizedSample:
        """Concatenate messages and adjust boundaries by the existing token length."""
        return TokenizedSample(
            ids=self.ids + other.ids,
            message_boundaries=self.message_boundaries
            + [end + len(self) for end in other.message_boundaries],
            roles=self.roles + other.roles,
        )

    def split(self) -> Iterator[TokenizedSample]:
        """Extract one document-question-option sequence per candidate."""
        state = self[:1]
        query = None
        for index, role in enumerate(self.roles):
            if role == "query":
                query = self[index : index + 1]
            elif role == "option":
                assert query is not None
                yield state + query + self[index : index + 1]


@dataclass
class Batch:
    """Padded candidate tensors for offline Transformers execution."""

    input_ids: torch.Tensor
    """Candidate token IDs with right padding."""
    attention_mask: torch.Tensor
    """One for real tokens and zero for padding."""

    @classmethod
    def from_inputs(cls, inputs: list[list[int]], pad_id: int) -> Batch:
        """Pad candidate sequences to the longest input in this submission."""
        length = max(map(len, inputs))
        return cls(
            torch.tensor([ids + [pad_id] * (length - len(ids)) for ids in inputs]),
            torch.tensor([[1] * len(ids) + [0] * (length - len(ids)) for ids in inputs]),
        )

    def to(self, device) -> Batch:
        """Transfer padded candidate tensors to the backbone input device."""
        return replace(
            self, input_ids=self.input_ids.to(device), attention_mask=self.attention_mask.to(device)
        )


@dataclass
class Encoded:
    """Candidate inputs and their document/question output grouping."""

    inputs: list[list[int]]
    """Unpadded candidate sequences in document, question, and option order."""
    requests: list[Request]
    """Original requests used to reconstruct nested results."""

    def results(self, scores: list[float]) -> list[list[Result]]:
        """Normalize candidate logits within each question and restore document order."""
        expected = sum(len(query.options or []) for item in self.requests for query in item.queries)
        if len(scores) != expected:
            message = "Engine returned a different number of candidate scores"
            raise ValueError(message)
        rows = iter(scores)
        documents = []
        for request in self.requests:
            answers = []
            for query in request.queries:
                options = query.options or []
                logits = torch.tensor([next(rows) for _ in options], dtype=torch.float32)
                probabilities = logits.softmax(0)
                answers.append(
                    Result(
                        choice=options[int(probabilities.argmax())],
                        probabilities=dict(zip(options, probabilities.tolist(), strict=True)),
                    )
                )
            documents.append(answers)
        return documents


class Step[T](ABC):
    """Pull items from a preparation stage.

    Examples:
        >>> FormatStep([]).next() is None
        True
    """

    @abstractmethod
    def next(self) -> T | None:
        """Return the next item, or None when exhausted."""
        ...


class FormatStep(Step[Sample]):
    """Pull requests and delegate message formatting to Sample.

    Args:
        requests: Documents to classify in output order.

    Examples:
        >>> FormatStep([]).next() is None
        True
    """

    def __init__(self, requests: list[Request]):
        """Initialize this call's request cursor."""
        self.requests = iter(requests)

    def next(self) -> Sample | None:
        """Format the next complete request."""
        request = next(self.requests, None)
        return None if request is None else Sample.from_request(request)


class TokenizeStep(Step[TokenizedSample]):
    """Delegate candidate tokenization to TokenizedSample.

    Args:
        parent: Formatted question source.
        tokenizer: Checkpoint tokenizer shared across calls.

    Examples:
        >>> isinstance(TokenizeStep, type)
        True
    """

    def __init__(self, parent: Step[Sample], tokenizer: PreTrainedTokenizerBase):
        """Attach the formatted source and tokenizer settings."""
        self.parent = parent
        self.tokenizer = tokenizer

    def next(self) -> TokenizedSample | None:
        """Pull and tokenize the next question's candidates."""
        sample = self.parent.next()
        if sample is None:
            return None
        return TokenizedSample.from_sample(sample, self.tokenizer)


class SplitStep(Step[TokenizedSample]):
    """Delegate candidate extraction to TokenizedSample.split().

    Args:
        parent: Tokenized complete-request source.

    Examples:
        >>> step = SplitStep(parent)  # doctest: +SKIP
        >>> candidate = step.next()  # doctest: +SKIP
    """

    def __init__(self, parent: Step[TokenizedSample]):
        """Attach the source and initialize this call's pending candidates."""
        self.parent = parent
        self.pending: Iterator[TokenizedSample] = iter(())

    def next(self) -> TokenizedSample | None:
        """Pull the next candidate, splitting a request when needed."""
        while (candidate := next(self.pending, None)) is None:
            sample = self.parent.next()
            if sample is None:
                return None
            self.pending = iter(sample.split())
        return candidate


class Processor:
    """Expose synchronous preparation and result decoding.

    Args:
        tokenizer: Checkpoint tokenizer with its trained chat template.

    Examples:
        >>> processor = Processor(tokenizer)  # doctest: +SKIP
        >>> processor.decode([], processor.encode([]))  # doctest: +SKIP
        []
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        """Store immutable preparation settings; each encode builds its own chain."""
        self.tokenizer = tokenizer

    @staticmethod
    def template() -> str:
        """Read the bundled linear-head template used when exporting checkpoints."""
        return files("social_jev").joinpath("templates/messages.jinja").read_text()

    def encode(self, requests: Request | list[Request]) -> Encoded:
        """Run a fresh Step.next chain and collect unpadded candidate sequences."""
        items = [requests] if isinstance(requests, Request) else requests
        formatted = FormatStep(items)
        tokenized = TokenizeStep(formatted, self.tokenizer)
        split = SplitStep(tokenized)
        inputs = []
        while (sample := split.next()) is not None:
            inputs.append(sample.ids)
        return Encoded(inputs, items)

    @staticmethod
    def decode(scores: list[float], encoded: Encoded) -> list[list[Result]]:
        """Delegate nested result reconstruction to the encoded dataclass."""
        return encoded.results(scores)
