# API reference

The public interface consists of `SocialJev` and the typed inputs and outputs `Query`, `Request`, and `Result`.

```python
from social_jev import Query, Request, Result, SocialJev
```

See [Concepts](concepts.md) for the request/result structure and [Inference](inference.md) for practical execution patterns.

## SocialJev.from_pretrained()

```text
SocialJev.from_pretrained(repo, *, backend="transformers", serve=False, **kwargs) -> SocialJev
```

Load a checkpoint and configure its inference backend.

| Argument | Meaning |
| --- | --- |
| `repo` | Hugging Face model repository ID or local checkpoint directory. |
| `backend` | `"transformers"` (default) or `"vllm"`. |
| `serve` | Launch a managed vLLM HTTP server. Requires `backend="vllm"`; defaults to `False`. |
| `**kwargs` | Additional keyword arguments forwarded to the selected backend, as described below. |

The checkpoint determines whether the trained head is linear or MLP. The tokenizer follows the model's `revision` when one is supplied.

```python
model = SocialJev.from_pretrained("social-jev/social-jev-0.6B-preview")
```

### Additional backend arguments

Beyond the explicit `repo`, `backend`, and `serve` arguments, additional keyword arguments are passed to the selected backend's loading or engine configuration. Their meaning and supported values are defined by that backend:

- **Transformers:** arguments go to the model's `from_pretrained()` loader. See the [Transformers model-loading reference](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained).
- **Local vLLM:** arguments go to the native engine configuration. See [vLLM engine arguments](https://docs.vllm.ai/en/v0.31.0/configuration/engine_args/).
- **Managed vLLM server:** keyword names are translated to native CLI options; for example, `tensor_parallel_size=2` becomes `--tensor-parallel-size 2`. See [vLLM server arguments](https://docs.vllm.ai/en/v0.31.0/cli/serve/).

Arguments must be appropriate for the selected backend. For example, `seed` is a vLLM engine option; it is not a SocialJev request field or a Transformers model-loading option.

SocialJev supplies the fixed integration needed to run its trained decision head through vLLM. It otherwise leaves native execution settings to the backend's defaults or the supplied arguments.

#### Example: Transformers precision

```python
model = SocialJev.from_pretrained(
    "social-jev/social-jev-0.6B-preview",
    dtype="bfloat16",
)
```

#### Example: vLLM engine settings

```python
model = SocialJev.from_pretrained(
    "social-jev/social-jev-0.6B-preview",
    backend="vllm",
    seed=0,
    max_model_len=8192,
    max_num_batched_tokens=32768,
)
```

The same native engine arguments can be supplied with `serve=True`.

#### Example: batch invariance for reproducibility

Batch invariance is a vLLM environment setting, not a `from_pretrained()` keyword argument. Set it before starting the inference script:

```bash
VLLM_BATCH_INVARIANT=1 uv run python inference.py
```

It reaches both local vLLM and the managed server. The engine seed already defaults to `0`; a fixed seed alone does not prevent numerical differences caused by batch composition or scheduling. Our method scores options rather than randomly sampling answers, so there is no request-level sampling seed to set.

vLLM's batch-invariance mode has hardware and operation restrictions and can affect performance. Exact end-to-end batch invariance of SocialJev's custom linear/MLP inference path has not yet been validated. See [vLLM batch invariance](https://docs.vllm.ai/en/v0.31.0/features/batch_invariance/), [vLLM reproducibility](https://docs.vllm.ai/en/v0.31.0/usage/reproducibility/), and the [inference guide's reproducibility section](inference.md#reproducibility).

## SocialJev.query()

```python
results = model.query(requests)
```

Submit requests synchronously and wait for their results. Available with Transformers, local vLLM, and managed-server vLLM.

**Arguments**

- `requests: Request | list[Request]`: one document request or a collection of document requests.

**Returns**

`list[list[Result]]`, ordered first by document, then by query for that document. A single `Request` also produces a nested list.

```python
first_query_result = results[0][0]
print(first_query_result.choice)
print(first_query_result.probabilities)
```

## SocialJev.aquery()

```python
results = await model.aquery(requests)
```

Submit requests asynchronously. Available with local and managed-server vLLM, including concurrent calls on the same model.

**Arguments**

- `requests: Request | list[Request]`: one document request or a collection of document requests.

**Returns**

Awaiting the call yields `list[list[Result]]` with the same document/query ordering as `.query()`. Each concurrent call returns results for its own inputs.

## SocialJev.stop()

```python
model.stop()
```

Shut down the vLLM engine workers and any managed server launched by the instance. Finish outstanding submissions first.

With Transformers, this releases the engine's model reference; calling it is not required for ordinary use.

## Query

```python
query = Query("Policy area?", options=["Transport", "Education"])
```

- `text: str`: the question defining the task, supplied positionally or as `text=...`.
- `options: list[str] | None`: a nonempty list of distinct candidate answers; defaults to `None`.

`Query("Policy area?")` is equivalent to `Query(text="Policy area?", options=None)`.
The query can be constructed without options, but inference currently requires them.

## Request

```python
request = Request(state="The bus network expanded.", queries=[query])
```

- `state: str`: the document text.
- `queries: list[Query]`: a nonempty list of queries about the document.

## Result

Each query returns a `Result` with:

- `choice: str`: the supplied candidate answer with the highest probability.
- `probabilities: dict[str, float]`: probabilities normalized over the supplied options, with keys in option order.

```python
result = results[0][0]
print(result.choice)
print(result.probabilities)
```
