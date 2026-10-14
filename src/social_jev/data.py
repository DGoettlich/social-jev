"""Dataclass transformations exposed through a pull-based Processor chain."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.resources import files
from typing import TYPE_CHECKING, cast

import torch
from transformers import AutoTokenizer

from .types import Request, Result

if TYPE_CHECKING:
    from collections.abc import Iterator

    from transformers import PreTrainedTokenizerBase


@dataclass
class Sample:
    """One document and question with its candidate answers."""

    state: str
    """Document being classified."""
    text: str
    """Question defining the classification task."""
    options: list[str]
    """Candidate answers in caller order."""

    @classmethod
    def from_request(cls, request: Request) -> list[Sample]:
        """Split a document into independently processed questions."""
        return [cls(request.state, query.text, query.options) for query in request.queries]

    def candidates(self) -> list[list[dict[str, str]]]:
        """Format one causal document-question-option sequence per candidate."""
        return [
            [
                {"role": "state", "content": self.state},
                {"role": "query", "content": self.text},
                {"role": "option", "content": option},
            ]
            for option in self.options
        ]


@dataclass
class TokenizedSample:
    """Unpadded candidate sequences for one question."""

    inputs: list[list[int]]
    """Candidate token sequences ending at the option closing token."""

    @classmethod
    def from_sample(cls, sample: Sample, tokenizer: PreTrainedTokenizerBase, maximum: int):
        """Tokenize candidate messages using the checkpoint's exact template."""
        inputs = [
            cast(
                "list[int]",
                tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False),
            )
            for messages in sample.candidates()
        ]
        if any(len(ids) > maximum for ids in inputs):
            message = "Candidate sequence exceeds model context capacity"
            raise ValueError(message)
        return cls(inputs)


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


@dataclass
class Encoded:
    """Candidate inputs and their document/question output grouping."""

    inputs: list[list[int]]
    """Unpadded candidate sequences in document, question, and option order."""
    requests: list[Request]
    """Original requests used to reconstruct nested results."""

    def results(self, scores: list[float]) -> list[list[Result]]:
        """Normalize candidate logits within each question and restore document order."""
        expected = sum(len(query.options) for item in self.requests for query in item.queries)
        if len(scores) != expected:
            message = "Engine returned a different number of candidate scores"
            raise ValueError(message)
        rows = iter(scores)
        documents = []
        for request in self.requests:
            answers = []
            for query in request.queries:
                logits = torch.tensor([next(rows) for _ in query.options], dtype=torch.float32)
                probabilities = logits.softmax(0)
                answers.append(
                    Result(
                        choice=query.options[int(probabilities.argmax())],
                        probabilities=dict(zip(query.options, probabilities.tolist(), strict=True)),
                    )
                )
            documents.append(answers)
        return documents


class Step[T](ABC):
    """Pull items from a preparation stage.

    Examples:
        >>> RequestStep([]).next() is None
        True
    """

    @abstractmethod
    def next(self) -> T | None:
        """Return the next item, or None when exhausted."""
        ...


class RequestStep(Step[Request]):
    """Read requests in their original order.

    Args:
        requests: Documents to classify.

    Examples:
        >>> RequestStep([]).next() is None
        True
    """

    def __init__(self, requests: list[Request]):
        """Initialize the request iterator."""
        self.requests = iter(requests)

    def next(self) -> Request | None:
        """Pull the next request."""
        return next(self.requests, None)


class FormatStep(Step[Sample]):
    """Delegate request formatting to Sample.

    Args:
        parent: Request source.

    Examples:
        >>> FormatStep(RequestStep([])).next() is None
        True
    """

    def __init__(self, parent: Step[Request]):
        """Attach the request source and initialize pending questions."""
        self.parent = parent
        self.pending: Iterator[Sample] = iter(())

    def next(self) -> Sample | None:
        """Pull a question, fetching its document when needed."""
        sample = next(self.pending, None)
        if sample is not None:
            return sample
        request = self.parent.next()
        if request is None:
            return None
        self.pending = iter(Sample.from_request(request))
        return next(self.pending)


class TokenizeStep(Step[TokenizedSample]):
    """Delegate candidate tokenization to TokenizedSample.

    Args:
        parent: Formatted question source.
        processor: Checkpoint tokenizer and context capacity.

    Examples:
        >>> isinstance(TokenizeStep, type)
        True
    """

    def __init__(self, parent: Step[Sample], processor: Processor):
        """Attach the formatted source and tokenizer settings."""
        self.parent = parent
        self.processor = processor

    def next(self) -> TokenizedSample | None:
        """Pull and tokenize the next question's candidates."""
        sample = self.parent.next()
        if sample is None:
            return None
        return TokenizedSample.from_sample(sample, self.processor.tokenizer, self.processor.maximum)


class Processor:
    """Expose synchronous preparation and result decoding.

    Args:
        tokenizer: Checkpoint tokenizer with its trained chat template.
        maximum: Maximum candidate sequence length.

    Examples:
        >>> processor = Processor(tokenizer, maximum=4096)  # doctest: +SKIP
        >>> processor.decode([], processor.encode([]))  # doctest: +SKIP
        []
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, maximum: int):
        """Store immutable preparation settings; each encode builds its own chain."""
        self.tokenizer = tokenizer
        self.maximum = maximum

    @classmethod
    def from_pretrained(cls, repo, maximum: int):
        """Load the checkpoint's tokenizer and template."""
        return cls(cast("PreTrainedTokenizerBase", AutoTokenizer.from_pretrained(repo)), maximum)

    @staticmethod
    def template() -> str:
        """Read the bundled linear-head template used when exporting checkpoints."""
        return files("social_jev").joinpath("templates/messages.jinja").read_text()

    def encode(self, requests: Request | list[Request]) -> Encoded:
        """Run a fresh Step.next chain and collect unpadded candidate sequences."""
        items = [requests] if isinstance(requests, Request) else requests
        source = RequestStep(items)
        formatted = FormatStep(source)
        tokenized = TokenizeStep(formatted, self)
        inputs = []
        while (sample := tokenized.next()) is not None:
            inputs.extend(sample.inputs)
        return Encoded(inputs, items)

    @staticmethod
    def decode(scores: list[float], encoded: Encoded) -> list[list[Result]]:
        """Delegate nested result reconstruction to the encoded dataclass."""
        return encoded.results(scores)
