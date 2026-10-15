# Async serving

`serve=True` starts a vLLM subprocess owned by SocialJev. Use `aquery()` for this
mode; `.query()` is for offline backends. Install the [vLLM extra](installation.md#optional-vllm-backend)
first.

## Start, submit and stop

```python
import asyncio
from social_jev import Query, Request, SocialJev


async def main():
    model = SocialJev.from_pretrained(
        "social-jev/social-jev-0.6B-preview",
        backend="vllm",
        serve=True,
        host="127.0.0.1",
        port=8000,
        gpu_memory_utilization=0.6,
        tensor_parallel_size=1,
    )
    try:
        results = await model.aquery(
            [
                Request(
                    state="Expand the rail network.",
                    queries=[Query(text="Policy area?", options=["Transport", "Education"])],
                ),
            ]
        )
        print(results[0][0].model_dump())
    finally:
        model.stop()


if __name__ == "__main__":
    asyncio.run(main())
```

Construction waits for server readiness and is synchronous. Start the model before
handling live traffic. Once started, `aquery()` moves encoding and decoding off the
caller's event-loop thread and awaits HTTP inference through a reusable client.

## Keep several batches in flight

Use the same model across submissions. Given a collection of request batches:

```python
semaphore = asyncio.Semaphore(4)


async def submit(batch):
    async with semaphore:
        return await model.aquery(batch)


results_by_batch = await asyncio.gather(*(submit(batch) for batch in batches))
```

The semaphore bounds active calls. Preparation for one batch can overlap inference
for another. `gather()` preserves submission order. For an unbounded stream, bound
queued work as well as active calls; do not accumulate one task per input forever.

## Server configuration and ownership

Native vLLM CLI settings pass through loading kwargs, including tensor and pipeline
parallelism, host, port, revision and scheduling settings. SocialJev fixes the
pooling integration and the served-model alias **`social-jev`**.

If host and port are omitted, vLLM's defaults apply. An all-interface bind can make
the server reachable from other machines; specify loopback explicitly when that is
what you want. Separate instances need separate ports.

The server uses the same Python environment as SocialJev. `.stop()` closes its HTTP
client and stops its owned server process group; await outstanding calls first.
Startup and shutdown have no additional SocialJev deadline.

Connecting this interface to an independently deployed server is not currently a
public capability. The examples above describe the server SocialJev starts itself.
