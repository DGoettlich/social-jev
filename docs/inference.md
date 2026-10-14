# Linear inference

The replacement package supports one method: a Qwen3 backbone followed by a
trained scalar linear head. Each candidate is encoded as document, question,
and one option. The head reads the option's final closing token. Raw candidate
scores are normalized together within each question.

## Execution

The base dependency set supports offline Transformers inference on CPU:

```python
from social_jev import Query, Request, SocialJev

request = Request(
    state="The council expanded the bus network.",
    queries=[Query(text="Policy area?", options=["Transport", "Education"])],
)
model = SocialJev.from_pretrained("<linear-checkpoint>")
try:
    results = model.query(request)
    print(results[0][0].choice)
finally:
    model.stop()
```

Both a single request and a list return `list[list[Result]]`, ordered by document
and then question. Empty input returns an empty list. Each query requires
nonempty distinct options. Few-shot examples are not part of this method.

On Linux with a supported CUDA GPU, the `vllm` extra adds vLLM 0.31.0 and HTTP
transport. Select offline execution with `backend="vllm"`; submit the whole
collection to `query()`. vLLM owns execution batching and prefix caching.
There is no public batch-size or engine-tuning interface.

For continuously supplied asynchronous work:

```python
import asyncio

model = SocialJev.from_pretrained(
    "<linear-checkpoint>",
    backend="vllm",
    serve=True,
)


async def classify(requests):
    semaphore = asyncio.Semaphore(8)

    async def submit(request):
        async with semaphore:
            return await model.aquery(request)

    try:
        return await asyncio.gather(*(submit(request) for request in requests))
    finally:
        model.stop()
```

Call `classify()` from your application event loop. Each `aquery()` moves
synchronous encoding and decoding to worker threads. HTTP uses a persistent
client on an owned internal loop, so synchronous `stop()` can close it even
when called inside the application's event loop. Finish or cancel and await
outstanding queries before stopping. The server binds only to localhost and
shutdown terminates only its owned process group. Start/load is synchronous;
initialize the model before starting concurrent submissions.

The caller owns dataset iteration and concurrency. Transformer submissions
must fit CPU memory; vLLM submissions must fit host memory and its context limit.
GPU memory utilization is fixed at 0.8, with one GPU and no chunked prefill.
This does not expose a remote service or provide automatic retries.

## Checkpoint contract

Export the trained model from `mert` into one self-contained HF repository:

- `config.json`: flat `SocialJevConfig`, derived from Qwen3 configuration, with
  `model_type="social_jev"` and `decision_head={"type": "linear"}`.
- SafeTensors weights: `backbone.*` using HF Qwen3 parameter names, plus
  `decision_head.weight` with shape `(1, hidden_size)` and
  `decision_head.bias` with shape `(1,)`.
- Tokenizer assets with the exact training template saved as `chat_template`.
  The bundled template is available through `Processor.template()` for export.

A trained backbone must be exported with its matching head. The exporter and
training dependencies remain in `mert`; no conversion or architecture-selection
machinery is included here. This replaces the LM-head checkpoint format.
The currently published 0.6B LM-head checkpoint is not a linear checkpoint.

`LinearModel` in `social_jev.transformers` owns HF serialization.
The vLLM plugin maps backbone weights to native Qwen3 loading and loads the
linear weight and bias directly into its replicated GPU pooler. It returns
only scalar logits, never full hidden-state vectors or vocabulary logits.

## Verification

The normal e2e suite loads a tiny real checkpoint and exercises inference,
candidate isolation, ordering, and the production HTTP/client lifecycle through
a CPU pooling fixture. That fixture does not verify the native GPU integration.

Run the native offline and online vLLM e2e on a Linux CUDA host with the extra:

```bash
SOCIAL_JEV_GPU_TESTS=1 uv run --extra vllm python -m pytest tests/e2e/test_vllm.py --no-cov
```
