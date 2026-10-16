"""Benchmark async SocialJev inference on FineWeb passages.

Run with ``uv run --group benchmark --extra vllm python scripts/fineweb.py --help``.
Dataset preparation and model startup are excluded from inference timing.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from itertools import batched
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import TYPE_CHECKING, Any, Self

import httpx
from datasets import load_dataset
from prometheus_client.parser import text_string_to_metric_families

from social_jev import Query, Request, SocialJev
from social_jev.data import Step

if TYPE_CHECKING:
    from collections.abc import Iterator

QUERY = Query(text="Does this passage discuss education policy?", options=["No", "Yes"])
"""Binary classification task shared by every passage."""
DATASET = "HuggingFaceFW/fineweb"
"""FineWeb source dataset."""
SUBSET = "sample-10BT"
"""FineWeb subset streamed for benchmark inputs."""
MAX_CHARS = 12_000
"""Maximum characters retained from each source document."""
MAX_MODEL_LEN = 8192
"""Maximum complete candidate length accepted by the benchmark server."""


@dataclass
class Document:
    """One source document from FineWeb."""

    text: str
    """Original document text."""

    def request(self) -> Request:
        """Slice the source text and attach the benchmark question."""
        return Request(state=self.text[:MAX_CHARS], queries=[QUERY])


class LoadStep(Step[Document]):
    """Stream FineWeb source documents.

    Examples:
        >>> document = LoadStep().next()  # doctest: +SKIP
    """

    def __init__(self):
        """Open the FineWeb sample as a stream."""
        self.rows = iter(load_dataset(DATASET, SUBSET, split="train", streaming=True))

    def next(self) -> Document | None:
        """Return the next source document, or None at exhaustion."""
        row = next(self.rows, None)
        return None if row is None else Document(row["text"])


class PassageStep(Step[Request]):
    """Pull documents and return sliced passage requests up to the limit.

    Examples:
        >>> step = PassageStep(LoadStep(), limit=100_000)  # doctest: +SKIP
        >>> request = step.next()  # doctest: +SKIP
    """

    def __init__(self, parent: Step[Document], limit: int):
        """Attach the source and initialize the passage count."""
        self.parent = parent
        self.limit = limit
        self.yielded = 0

    def next(self) -> Request | None:
        """Return the next sliced passage, or None at the limit or source end."""
        if self.yielded >= self.limit:
            return None
        document = self.parent.next()
        if document is None:
            return None
        self.yielded += 1
        return document.request()


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


METRICS = {
    "input_tokens": "vllm:prompt_tokens_total",
    "finished": "vllm:request_success_total",
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "cache_usage": "vllm:kv_cache_usage_perc",
    "preemptions": "vllm:num_preemptions_total",
}
"""vLLM 0.31.0 counters and gauges included in the server trace."""


class Telemetry:
    """Continuously record server metrics and the selected GPU's driver statistics.

    Examples:
        >>> async with Telemetry(model.engine.url, output) as monitor:  # doctest: +SKIP
        ...     results = await benchmark.run(requests)
    """

    def __init__(self, server_url: str, output: Path):
        """Configure trace paths and the single GPU selected by the launcher."""
        self.server_url = server_url
        self.gpu_path = output.parent / "gpu.csv"
        self.server_path = output.parent / "server.jsonl"
        self.gpu_id = os.environ["CUDA_VISIBLE_DEVICES"]
        self.stopping = asyncio.Event()
        self.samples: list[dict[str, float]] = []
        self.markers: dict[str, float] = {}

    async def __aenter__(self) -> Self:
        """Start a persistent GPU sampler and periodic server collection."""
        info_process = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--id=" + self.gpu_id,
            "--query-gpu=uuid,name,memory.total",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
        )
        info, _ = await info_process.communicate()
        if info_process.returncode != 0:
            message = "nvidia-smi could not query the selected GPU"
            raise RuntimeError(message)
        self.gpu_info = info.decode().strip()
        self.gpu_file = self.gpu_path.open("w")
        self.server_file = self.server_path.open("w")
        self.client = httpx.AsyncClient(base_url=self.server_url, timeout=10)
        self.gpu_process = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--id=" + self.gpu_id,
            "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,nounits",
            "--loop-ms=1000",
            stdout=self.gpu_file,
            env={**os.environ, "TZ": "UTC"},
        )
        self.task = asyncio.create_task(self.sample_server())
        return self

    async def __aexit__(self, *exc) -> None:
        """Stop both samplers and close their files, including on inference errors."""
        self.stopping.set()
        try:
            await self.task
        finally:
            self.gpu_process.terminate()
            await self.gpu_process.wait()
            self.gpu_file.close()
            self.server_file.close()
            await self.client.aclose()

    async def snapshot(self) -> dict[str, float]:
        """Read cumulative counters and current scheduler gauges from vLLM."""
        response = await self.client.get("/metrics")
        response.raise_for_status()
        samples = [
            sample
            for family in text_string_to_metric_families(response.text)
            for sample in family.samples
        ]
        values = {
            key: sum(sample.value for sample in samples if sample.name == name)
            for key, name in METRICS.items()
        }
        for key, source in [
            ("computed_tokens", "local_compute"),
            ("cached_tokens", "local_cache_hit"),
        ]:
            values[key] = sum(
                sample.value
                for sample in samples
                if sample.name == "vllm:prompt_tokens_by_source_total"
                and sample.labels["source"] == source
            )
        values["timestamp"] = datetime.now(UTC).timestamp()
        return values

    async def sample_server(self) -> None:
        """Append timestamped scheduler and token measurements once per second."""
        while not self.stopping.is_set():
            sample = await self.snapshot()
            self.samples.append(sample)
            self.server_file.write(json.dumps(sample) + "\n")
            self.server_file.flush()
            await asyncio.sleep(1)

    async def wait_completed(self, expected: float) -> dict[str, float]:
        """Wait outside inference timing for metrics to include completed candidates."""
        async with asyncio.timeout(30):
            while (sample := await self.snapshot())["finished"] < expected:
                await asyncio.sleep(0.2)
        self.server_file.write(json.dumps({"event": "counters", **sample}) + "\n")
        self.server_file.flush()
        return sample

    def mark(self, event: str) -> None:
        """Record timed boundaries so summaries exclude warmup and metric drain."""
        self.markers[event] = datetime.now(UTC).timestamp()
        self.server_file.write(
            json.dumps({"event": event, "timestamp": self.markers[event]}) + "\n"
        )
        self.server_file.flush()

    def report(self, before: dict[str, float], after: dict[str, float], elapsed: float) -> dict:
        """Summarize token deltas and samples falling inside the inference interval."""
        report: dict[str, float] = {
            key: int(after[key] - before[key])
            for key in ["input_tokens", "computed_tokens", "cached_tokens", "preemptions"]
        }
        report.update(
            {
                key + "_per_second": report[key] / elapsed
                for key in ["input_tokens", "computed_tokens"]
            }
        )
        with self.gpu_path.open() as handle:
            gpu_samples = [
                {key.strip().split(" [")[0]: value.strip() for key, value in row.items()}
                for row in csv.DictReader(handle)
                if self.markers["start"]
                <= datetime.strptime(row["timestamp"].strip(), "%Y/%m/%d %H:%M:%S.%f")
                .replace(tzinfo=UTC)
                .timestamp()
                <= self.markers["end"]
            ]
        server_samples = [
            sample
            for sample in self.samples
            if self.markers["start"] <= sample["timestamp"] <= self.markers["end"]
        ]
        report["gpu_samples"] = len(gpu_samples)
        report["server_samples"] = len(server_samples)
        if gpu_samples:
            report.update(
                {
                    "gpu_utilization_mean_percent": mean(
                        float(row["utilization.gpu"]) for row in gpu_samples
                    ),
                    "gpu_memory_peak_mib": max(float(row["memory.used"]) for row in gpu_samples),
                    "gpu_power_mean_watts": mean(float(row["power.draw"]) for row in gpu_samples),
                }
            )
        if server_samples:
            report.update(
                {
                    "running_requests_mean": mean(sample["running"] for sample in server_samples),
                    "waiting_requests_peak": max(sample["waiting"] for sample in server_samples),
                    "kv_cache_usage_peak_fraction": max(
                        sample["cache_usage"] for sample in server_samples
                    ),
                }
            )
        return report


def candidate_lengths(model: SocialJev, requests: list[Request]) -> list[int]:
    """Measure complete candidates before timing and reject oversized passages."""
    lengths = [
        len(ids)
        for batch in batched(requests, 128)
        for ids in model.processor.encode(list(batch)).inputs
    ]
    if max(lengths) > MAX_MODEL_LEN:
        message = f"Longest candidate is {max(lengths)} tokens; max_model_len is {MAX_MODEL_LEN}"
        raise ValueError(message)
    return lengths


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
    output = args.output or Path("artifacts") / "fineweb" / model_name / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    pipeline = PassageStep(LoadStep(), limit=args.documents)
    requests = list(iter(pipeline.next, None))
    print(f"Prepared {len(requests)} passages (requested {args.documents}).", flush=True)
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
        lengths = candidate_lengths(model, requests)
        await model.aquery(
            Request(
                state="This separate warmup passage discusses funding for schools.",
                queries=[QUERY],
            )
        )
        async with Telemetry(model.engine.url, output) as telemetry:
            before = await telemetry.wait_completed(len(QUERY.options))
            telemetry.mark("start")
            results = await SpeedBench(model, args.batch_size, args.concurrency).run(requests)
            telemetry.mark("end")
            after = await telemetry.wait_completed(before["finished"] + len(lengths))
        results.update(telemetry.report(before, after, results["elapsed_seconds"]))
        results["startup_seconds"] = startup_seconds
        report = {
            "params": {
                "model": args.model,
                "documents": len(requests),
                "dataset": DATASET,
                "subset": SUBSET,
                "max_chars": MAX_CHARS,
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
