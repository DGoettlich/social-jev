# API reference

Import the public interface from `social_jev`. Model loading accepts an HF repository
ID or local checkpoint directory.

## Loading and execution

| Argument | Meaning |
| --- | --- |
| `repo` | HF model repository or local directory. |
| `backend="transformers"` | Offline Transformers; `"vllm"` selects vLLM. |
| `serve=False` | Set `True` with vLLM to use async serving. |
| `**kwargs` | Native model-loading or vLLM engine/CLI settings. |

The tokenizer uses the model's `revision`. Linear versus MLP is determined by the
checkpoint rather than a loading flag. All query methods return nested results.

::: social_jev.SocialJev
    options:
      members: [from_pretrained, query, aquery, stop]

## Request types

::: social_jev.Request
    options:
      members: [state, queries]

::: social_jev.Query
    options:
      members: [text, options]

## Result type

::: social_jev.Result
    options:
      members: [choice, probabilities]

## Processor

`model.processor` exposes request encoding and result decoding. Most applications
should use the query methods, which pair processing with the selected backend.

::: social_jev.Processor
    options:
      members: [encode, decode]
