"""Benchmark async SocialJev inference on Manifesto passages.

Run with ``uv run --group benchmark --extra vllm python scripts/speedbench.py --help``.
Dataset preparation and model startup are excluded from inference timing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from dataclasses import dataclass, field
from importlib.metadata import version
from itertools import batched, groupby
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING, Any

from datasets import load_dataset

from social_jev import Query, Request, SocialJev
from social_jev.data import Step

if TYPE_CHECKING:
    from collections.abc import Iterator

QUERY = Query(text="Does this passage discuss education policy?", options=["No", "Yes"])
"""Binary classification task shared by every passage."""
DATASET = "social-jev/manifesto"
"""Manifesto source dataset."""
REVISION = "v0.1alpha"
"""Fixed dataset revision for reproducible input preparation."""


@dataclass
class Document:
    """Manifesto source units belonging to one document."""

    id: str
    """Source document identifier."""
    units: list[dict[str, Any]]
    """Source rows in their original order."""

    def passages(self) -> Iterator[Passage]:
        """Group consecutive units using mert's labelling rules."""
        rng = random.Random(f"group:{self.id}")
        start = 0
        while start < len(self.units):
            end = min(start + rng.randint(5, 40), len(self.units))
            units = self.units[start:end]
            start = end
            if any(unit["labels"] for unit in units):
                yield Passage(units)


@dataclass
class Passage:
    """Consecutive source units forming one inference input."""

    units: list[dict[str, Any]]
    """Rows included in this passage."""

    def request(self) -> Request:
        """Join source texts and attach the benchmark question."""
        return Request(state="\n".join(unit["text"] for unit in self.units), queries=[QUERY])


class LoadStep(Step[Document]):
    """Stream original Manifesto rows grouped by source document.

    Examples:
        >>> step = LoadStep()  # doctest: +SKIP
        >>> document = step.next()  # doctest: +SKIP
    """

    def __init__(self):
        """Open the pinned original training split."""
        rows = load_dataset(DATASET, "original", revision=REVISION, split="train", streaming=True)
        self.documents = groupby(rows, key=lambda row: row["document_id"])

    def next(self) -> Document | None:
        """Read the next source document, or None when exhausted."""
        item = next(self.documents, None)
        return None if item is None else Document(item[0], list(item[1]))


class PassageStep(Step[Request]):
    """Pull documents and return passage requests up to the limit.

    Examples:
        >>> step = PassageStep(LoadStep(), limit=100)  # doctest: +SKIP
        >>> request = step.next()  # doctest: +SKIP
    """

    def __init__(self, parent: Step[Document], limit: int):
        """Attach the source and initialize this run's passage cursor."""
        self.parent = parent
        self.limit = limit
        self.yielded = 0
        self.pending: Iterator[Passage] = iter(())

    def next(self) -> Request | None:
        """Return the next request, or None at the limit or source end."""
        if self.yielded >= self.limit:
            return None
        while (passage := next(self.pending, None)) is None:
            document = self.parent.next()
            if document is None:
                return None
            self.pending = document.passages()
        self.yielded += 1
        return passage.request()


@dataclass
class Measurements:
    """Completed classifications and per-call execution latencies."""

    classifications: int = 0
    """Number of returned question results."""
    latencies: list[float] = field(default_factory=list)
    """Call durations including processing, inference and decoding."""

    def report(self, documents: int, elapsed: float) -> dict[str, Any]:
        """Summarize whole-run throughput and nearest-rank batch latency."""
        ordered = sorted(self.latencies)
        return {
            "documents": documents,
            "classifications": self.classifications,
            "elapsed_seconds": elapsed,
            "documents_per_second": documents / elapsed,
            "classifications_per_second": self.classifications / elapsed,
            "batch_latency_median_seconds": median(ordered),
            "batch_latency_p95_seconds": ordered[(95 * len(ordered) + 99) // 100 - 1],
        }


class SpeedBench:
    """Measure async throughput with a fixed number of submission workers.

    Examples:
        >>> bench = SpeedBench(model, batch_size=128, concurrency=16)  # doctest: +SKIP
        >>> report = asyncio.run(bench.run(requests))  # doctest: +SKIP
    """

    def __init__(self, model: SocialJev, batch_size: int, concurrency: int):
        """Configure the loaded model and client submission sizes."""
        self.model = model
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.measurements = Measurements()

    async def consume(self, batches: Iterator[tuple[Request, ...]]) -> None:
        """Submit successive batches through one concurrent worker."""
        for batch in batches:
            started = perf_counter()
            results = await self.model.aquery(list(batch))
            self.measurements.latencies.append(perf_counter() - started)
            self.measurements.classifications += sum(len(row) for row in results)

    async def run(self, requests: list[Request]) -> dict[str, Any]:
        """Process the complete collection and report end-to-end throughput."""
        self.measurements = Measurements()
        batches = batched(requests, self.batch_size)
        started = perf_counter()
        await asyncio.gather(*(self.consume(batches) for _ in range(self.concurrency)))
        return self.measurements.report(len(requests), perf_counter() - started)


def arguments() -> argparse.Namespace:
    """Read one benchmark configuration from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="social-jev/social-jev-0.6B-preview")
    parser.add_argument("--documents", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


async def main() -> None:
    """Prepare passages, warm a fresh server and save one timed benchmark."""
    args = arguments()
    pipeline = PassageStep(LoadStep(), limit=args.documents)
    requests = list(iter(pipeline.next, None))
    print(f"Prepared {len(requests)} passages (requested {args.documents}).", flush=True)
    started = perf_counter()
    model = SocialJev.from_pretrained(
        args.model,
        backend="vllm",
        serve=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    startup_seconds = perf_counter() - started
    try:
        await model.aquery(
            Request(
                state="This separate warmup passage discusses funding for schools.",
                queries=[QUERY],
            )
        )
        report = await SpeedBench(model, args.batch_size, args.concurrency).run(requests)
        report.update(
            {
                "model": args.model,
                "dataset": DATASET,
                "dataset_revision": REVISION,
                "requested_documents": args.documents,
                "batch_size": args.batch_size,
                "concurrency": args.concurrency,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "startup_seconds": startup_seconds,
                "query": QUERY.model_dump(),
                "versions": {name: version(name) for name in ("social-jev", "vllm", "datasets")},
            }
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        model.stop()


if __name__ == "__main__":
    asyncio.run(main())
