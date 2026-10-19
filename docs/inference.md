# Inference

This page explains how to run a SocialJev workload efficiently and reproducibly: choosing a submission pattern, processing requests, and configuring model execution. The [getting-started guide](index.md) introduces model loading, and [Concepts](concepts.md) explains the typed inputs and outputs.

## Choose a submission pattern

A prepared collection that fits into host memory can be submitted in one synchronous `.query()` call. This lets vLLM schedule the work internally while the application waits for the results.

When inputs or results are too large to retain together, successive `.query()` calls can process smaller portions sequentially. Asynchronous requests can be more efficient because they allow preparation of one batch to overlap with inference on another, while multiple batches remain in progress.

These are choices about how the application proceeds. `.aquery()` enables concurrency; it does not automatically make every workload faster. The appropriate submission size and number of outstanding calls depend on input lengths, available memory, and the surrounding processing.

For sustained workloads, the package provides an optional vLLM integration:

```bash
uv pip install 'social-jev[vllm]'
```

The examples below use local vLLM. Both `.query()` and `.aquery()` also work with the managed server described under [Configure execution](#configure-execution).

## Synchronous submissions

`.query()` accepts one request or a list of requests and waits for their results. For a prepared collection, the call is simply:

```python
results = model.query(requests)
```

A complete small example:

```python
from social_jev import Query, Request, SocialJev


def main():
    query = Query(
        text="Which policy area does this document concern?",
        options=["Transport", "Education", "Healthcare"],
    )
    requests = [
        Request(state="The council expanded the bus network.", queries=[query]),
        Request(state="The government recruited more teachers.", queries=[query]),
    ]
    model = SocialJev.from_pretrained(
        "social-jev/social-jev-0.6B-preview",
        backend="vllm",
    )
    try:
        results = model.query(requests)
        print(results[0][0].choice)
        print(results[1][0].choice)
    finally:
        model.stop()


if __name__ == "__main__":
    main()
```

The returned results retain document order, then query order for each document.

### Submission size and memory

With vLLM, the collection passed to `.query()` is **not** one simultaneous GPU batch. vLLM schedules the resulting token sequences internally. The requests, their prepared inputs, and the returned results still consume host memory, so submitting everything at once is appropriate only when those fit.

Smaller successive calls allow results to be consumed before submitting more input. This remains synchronous inference; it does not require `.aquery()`.

With the default Transformers backend, each call instead pads its sequences into one model batch. Its submission size therefore directly affects device-memory requirements as well as host-memory use.

## Concurrent asynchronous submissions

`.aquery()` allows the application to continue other asynchronous work while awaiting inference. Several calls can share one loaded vLLM model, allowing preparation and result handling to overlap with execution.

The following small script illustrates overlapping calls. Each call submits one document, but either could also receive a list of requests:

```python
import asyncio

from social_jev import Query, Request, SocialJev


async def main():
    query = Query(
        text="Which policy area does this document concern?",
        options=["Transport", "Education", "Healthcare"],
    )
    first_request = Request(
        state="The council expanded the bus network.",
        queries=[query],
    )
    second_request = Request(
        state="The government recruited more teachers.",
        queries=[query],
    )
    model = SocialJev.from_pretrained(
        "social-jev/social-jev-0.6B-preview",
        backend="vllm",
    )
    try:
        first, second = await asyncio.gather(
            model.aquery(first_request),
            model.aquery(second_request),
        )
        print(first[0][0].choice)
        print(second[0][0].choice)
    finally:
        model.stop()


if __name__ == "__main__":
    asyncio.run(main())
```

Each call returns results for its own inputs. In this example, `gather()` associates the first returned value with the first submission and the second with the second submission, regardless of which completes first.

### Batch size and concurrency

**Batch size** is the number of documents included in one submission. **Concurrency** is the number of submissions allowed to remain in progress together. For example, 16 active calls containing 128 documents each allow up to 2,048 documents in flight. These values illustrate the relationship, rather than an optimal configuration.

Neither quantity is the GPU batch size. vLLM batches the submitted sequences internally, and each document creates one sequence per query–option combination. Input length, query count, and option count therefore affect how much work a submission represents.

For an incremental workload, the application controls when inputs are loaded, how many calls remain outstanding, and when completed results are saved or released. There is no requirement to load all batches first or accumulate all results. The small two-call example above is an illustration of concurrency, not a prescribed dataset-processing architecture.

SocialJev moves its own encoding and decoding off the caller's event-loop thread. Other loading or processing performed by the application needs to be arranged so it can overlap inference too.

The repository's [Wikipedia](https://github.com/social-jev/social-jev/blob/main/scripts/wikipedia.py) and [FineWeb](https://github.com/social-jev/social-jev/blob/main/scripts/fineweb.py) runs provide concrete references for larger concurrent workloads. They include benchmark instrumentation and prepare their input collections before timed inference; they are not streaming-inference examples.

## Configure execution

Native vLLM options pass through `from_pretrained()`. For example:

```python
model = SocialJev.from_pretrained(
    "social-jev/social-jev-0.6B-preview",
    backend="vllm",
    max_model_len=8192,
    max_num_batched_tokens=32768,
    gpu_memory_utilization=0.9,
)
```

These values are an example configuration, not package defaults.

### Context length

`max_model_len` must accommodate a complete document–query–option sequence, including the chat-template tokens. SocialJev does not automatically truncate or split oversized documents. Splitting a document into independently scored passages changes the task and belongs in the application's input preparation.

### GPU memory and scheduling

`gpu_memory_utilization` controls vLLM's memory allocation budget. It is not a strict cap on every allocation in the process.

`max_num_batched_tokens` controls vLLM's token budget per scheduling step. It does not limit how many documents the package accepts in a call. Unspecified performance settings, including prefix caching and chunked prefill, use vLLM's defaults.

### Multiple GPUs

Native options such as `tensor_parallel_size` can distribute model execution across GPUs:

```python
model = SocialJev.from_pretrained(
    "social-jev/social-jev-0.6B-preview",
    backend="vllm",
    tensor_parallel_size=2,
)
```

This is a vLLM model-execution setting, independent of the number of concurrent Python submissions. Multi-GPU execution through SocialJev has not yet been validated in our smoke checks.

### Local execution and a managed server

Local vLLM supports both synchronous and concurrent asynchronous calls. It can handle complete collections or requests prepared incrementally.

`serve=True` instead launches a managed vLLM server, with the same two submission methods:

```python
model = SocialJev.from_pretrained(
    "social-jev/social-jev-0.6B-preview",
    backend="vllm",
    serve=True,
)
```

A persistent server is useful when separate applications or clients share one loaded model. The package currently launches and owns its server; attaching another SocialJev client to an already-running server is not yet exposed. Dataset size and concurrency alone are not reasons to require `serve=True`.

After outstanding submissions finish, `model.stop()` shuts down the vLLM workers and any managed server. Explicit shutdown is not required for the Transformers backend.

The [API reference](api.md) lists the loading arguments and submission methods.

## Reproducibility

SocialJev scores the supplied options and returns their probabilities and highest-scoring choice. It does not randomly sample an answer, so there is no request-level sampling seed to set.

Nevertheless, floating-point scores can change with batch composition and execution scheduling. A fixed random seed alone does not prevent those differences; sufficiently close scores can change the selected option.

### Engine seed and batch invariance

The engine seed can be supplied through native vLLM arguments and already defaults to `0`:

```python
model = SocialJev.from_pretrained(
    "social-jev/social-jev-0.6B-preview",
    backend="vllm",
    seed=0,
)
```

vLLM provides a batch-invariance mode intended to make results independent of the requests processed alongside them. Set its environment variable before starting the inference script:

```bash
VLLM_BATCH_INVARIANT=1 uv run python inference.py
```

The setting reaches both local vLLM and a managed server. It has hardware and operation restrictions, remains a beta feature, and can affect performance. See [vLLM's batch-invariance documentation](https://docs.vllm.ai/en/v0.31.0/features/batch_invariance/).

### What is currently established

Our backend checks establish agreement within numerical tolerance, not an exact reproducibility guarantee across batch sizes or concurrent schedules. End-to-end batch invariance of the SocialJev backbone and custom linear/MLP head has not yet been validated.

For a reproducible run, retain the requests and supplied options, select a fixed model revision with `revision`, and record the package/software versions, hardware, precision, and parallelism settings. vLLM's reproducibility scope is the same hardware and vLLM version; it does not promise identical results across different environments. See [vLLM's reproducibility guidance](https://docs.vllm.ai/en/v0.31.0/usage/reproducibility/).
