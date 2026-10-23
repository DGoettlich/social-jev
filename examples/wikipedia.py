"""Benchmark Wikipedia chunks through SocialJev's async inference interface."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import deque
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING, cast

from datasets import load_dataset
from fineweb import MAX_MODEL_LEN, QUERY, SpeedBench, Telemetry, candidate_lengths

from social_jev import Request, SocialJev
from social_jev.data import Step

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

DATASET = "omarkamali/wikipedia-monthly"
"""Wikipedia article repository."""
SNAPSHOT = "20261001"
"""Fixed Wikipedia dump date."""
SHARDS = 1000
"""Number of randomly selected English shards."""
CHUNK_TOKENS = 4096
"""Document tokens per chunk, excluding the classification prompt."""


@dataclass
class Document:
    """One complete Wikipedia article."""

    text: str
    """Article text."""

    def tokenize(self, tokenizer: PreTrainedTokenizerBase) -> TokenizedDocument:
        """Encode article text without special tokens."""
        return TokenizedDocument(tokenizer.encode(self.text, add_special_tokens=False))


@dataclass
class TokenizedDocument:
    """An article or chunk represented by token IDs."""

    ids: list[int]
    """Document token IDs."""

    def split(self) -> list[TokenizedDocument]:
        """Slice the document into consecutive chunks, retaining the remainder."""
        return [
            TokenizedDocument(self.ids[start : start + CHUNK_TOKENS])
            for start in range(0, len(self.ids), CHUNK_TOKENS)
        ]

    def into_passage(self, tokenizer: PreTrainedTokenizerBase) -> Request:
        """Decode the chunk and attach the benchmark question."""
        return Request(
            state=cast("str", tokenizer.decode(self.ids, clean_up_tokenization_spaces=False)),
            queries=[QUERY],
        )


class LoadStep(Step[Document]):
    """Stream shuffled articles from 1,000 randomly selected shards.

    Examples:
        >>> source = LoadStep(limit=100_000)  # doctest: +SKIP
        >>> document = source.next()  # doctest: +SKIP
    """

    def __init__(self, limit: int):
        """Select fixed random shards and initialize the shuffled stream."""
        self.shards = Random(42).sample(range(1450), k=SHARDS)
        base = f"hf://datasets/{DATASET}/{SNAPSHOT}/en/train"
        dataset = load_dataset(
            "parquet",
            data_files=[f"{base}/train_part_{index:03d}.parquet" for index in self.shards],
            split="train",
            streaming=True,
        )
        self.rows = iter(dataset.shuffle(seed=42, buffer_size=10_000))
        self.documents = 0
        self.limit = limit

    def next(self) -> Document | None:
        """Return the next article, or None at exhaustion."""
        if self.documents >= self.limit:
            return None
        row = next(self.rows, None)
        if row is None:
            return None
        self.documents += 1
        return Document(row["text"])


class TokenizeStep(Step[TokenizedDocument]):
    """Tokenize complete articles pulled from the source.

    Examples:
        >>> tokens = TokenizeStep(source, tokenizer)  # doctest: +SKIP
        >>> document = tokens.next()  # doctest: +SKIP
    """

    def __init__(self, parent: Step[Document], tokenizer: PreTrainedTokenizerBase):
        """Connect the document source and model tokenizer."""
        self.parent = parent
        self.tokenizer = tokenizer

    def next(self) -> TokenizedDocument | None:
        """Pull and tokenize one article."""
        document = self.parent.next()
        return None if document is None else document.tokenize(self.tokenizer)


class SplitStep(Step[TokenizedDocument]):
    """Drain buffered article chunks one at a time.

    Examples:
        >>> chunks = SplitStep(tokens)  # doctest: +SKIP
        >>> chunk = chunks.next()  # doctest: +SKIP
    """

    def __init__(self, parent: Step[TokenizedDocument]):
        """Connect the tokenized source and initialize its chunk buffer."""
        self.parent = parent
        self.buffer: deque[TokenizedDocument] = deque()

    def next(self) -> TokenizedDocument | None:
        """Refill the empty buffer and return its next chunk."""
        if not self.buffer:
            document = self.parent.next()
            if document is None:
                return None
            self.buffer.extend(document.split())
        return self.buffer.popleft()


class IntoPassageStep(Step[Request]):
    """Convert every article chunk into a classification request.

    Examples:
        >>> passages = IntoPassageStep(chunks, tokenizer)  # doctest: +SKIP
        >>> request = passages.next()  # doctest: +SKIP
    """

    def __init__(self, parent: Step[TokenizedDocument], tokenizer: PreTrainedTokenizerBase):
        """Connect the chunk source and model tokenizer."""
        self.parent = parent
        self.tokenizer = tokenizer

    def next(self) -> Request | None:
        """Return one chunk request, or None at source exhaustion."""
        chunk = self.parent.next()
        if chunk is None:
            return None
        return chunk.into_passage(self.tokenizer)


def arguments() -> argparse.Namespace:
    """Read one benchmark configuration from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="social-jev/social-jev-0.6B-preview")
    parser.add_argument("--documents", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


async def main() -> None:
    """Prepare passages, monitor one timed run and save its parameters and results."""
    args = arguments()
    model_name = args.model.rsplit("/", 1)[-1].replace("social-jev", "socialjev").lower()
    output = args.output or Path("artifacts") / "wikipedia" / model_name / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    model = SocialJev.from_pretrained(
        args.model,
        backend="vllm",
        serve=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=args.max_num_batched_tokens,
        port=args.port,
    )
    startup_seconds = perf_counter() - started
    try:
        print(
            f"Preparing {args.documents} source articles from {SHARDS} selected shards.", flush=True
        )
        source = LoadStep(limit=args.documents)
        tokens = TokenizeStep(source, model.processor.tokenizer)
        chunks = SplitStep(tokens)
        pipeline = IntoPassageStep(chunks, model.processor.tokenizer)
        requests = list(iter(pipeline.next, None))
        print(f"Prepared {len(requests)} chunks from {source.documents} articles.", flush=True)
        lengths = candidate_lengths(model, requests)
        await model.aquery(
            Request(
                state="This separate warmup passage discusses funding for schools.",
                queries=[QUERY],
            )
        )
        async with Telemetry(model.engine.url, output) as telemetry:
            assert QUERY.options is not None
            before = await telemetry.wait_completed(len(QUERY.options))
            telemetry.mark("start")
            results = await SpeedBench(model, args.batch_size, args.concurrency).run(requests)
            telemetry.mark("end")
            after = await telemetry.wait_completed(before["finished"] + len(lengths))
        results.update(telemetry.report(before, after, results["elapsed_seconds"]))
        results["chunks_per_second"] = results.pop("documents_per_second")
        results["startup_seconds"] = startup_seconds
        report = {
            "params": {
                "model": args.model,
                "chunks": len(requests),
                "source_articles": source.documents,
                "document_limit": args.documents,
                "dataset": DATASET,
                "snapshot": SNAPSHOT,
                "language": "en",
                "shards": source.shards,
                "shuffle_buffer": 10_000,
                "chunk_tokens": CHUNK_TOKENS,
                "query": QUERY.model_dump(),
                "batch_size": args.batch_size,
                "concurrency": args.concurrency,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_model_len": MAX_MODEL_LEN,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "port": args.port,
                "telemetry_interval_seconds": 1,
                "input_sequence_tokens": {
                    "min": min(lengths),
                    "median": median(lengths),
                    "max": max(lengths),
                    "total": sum(lengths),
                },
                "versions": {name: version(name) for name in ("social-jev", "vllm")},
                "gpu": telemetry.gpu_info,
            },
            "results": results,
        }
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        model.stop()


if __name__ == "__main__":
    asyncio.run(main())
