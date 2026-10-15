# Batch inference

Offline inference takes a complete collection of requests and returns results in
that same order. No application batching loop is required for vLLM.

## Submit the whole collection

```python
from social_jev import Query, Request, SocialJev

query = Query(text="Policy area?", options=["Transport", "Education"])
requests = [
    Request(state="Expand the rail network.", queries=[query]),
    Request(state="Build more classrooms.", queries=[query]),
]
model = SocialJev.from_pretrained("social-jev/social-jev-0.6B-preview")
try:
    results = model.query(requests)
finally:
    model.stop()
```

## Use vLLM

With the [optional extra](installation.md#optional-vllm-backend) installed, change
only model loading:

```python
model = SocialJev.from_pretrained(
    "social-jev/social-jev-0.6B-preview",
    backend="vllm",
    gpu_memory_utilization=0.6,
    tensor_parallel_size=1,
)
try:
    results = model.query(requests)
finally:
    model.stop()
```

SocialJev submits all candidate sequences to vLLM once. vLLM schedules internal
batches and can reuse document prefixes across candidate answers. Its
`max_num_batched_tokens` setting controls the token budget per scheduling step,
not how many requests you may pass to `.query()`.

Architecture and causal LAST pooling are fixed by the classification method.
Performance settings otherwise use native vLLM defaults or your overrides. Memory
utilization is a native allocation budget, not a strict process-wide memory cap.

## Place the Transformers model on a device

Native Transformers keyword arguments, such as `dtype`, reach model loading.
For a simple CUDA placement without adding Accelerate:

```python
import torch

model = SocialJev.from_pretrained(
    "social-jev/social-jev-0.6B-preview",
    dtype=torch.bfloat16,
)
model.engine.model.to("cuda")
```

This accesses the exposed engine; SocialJev does not currently offer a separate
public device-placement method. Native `device_map` loading requires Accelerate
in your environment. The engine and processor share one tokenizer, whose saved
padding ID is used for unequal-length candidates.

## Size a submission

Transformers pads all candidates in a `.query()` call into one batch. Very large
collections can therefore use substantial memory; split them in application code
when using that backend. Offline vLLM schedules the collection internally, but
requests, token IDs and results still need to fit in host memory.

Inputs beyond the model context limit are not automatically truncated. For input
that arrives over time, use [async serving](serving.md).
