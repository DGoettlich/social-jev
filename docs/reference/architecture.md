# Pipeline and model

SocialJev implements classification inference, not training or evaluation tooling.
Its checkpoint combines a causal Qwen3 backbone with a scalar linear or MLP head.

## Request processing

```text
Request(s)
  → Sample.messages
  → TokenizedSample
  → split candidate sequences
  → backend scores
  → Encoded.results()
  → list[list[Result]]
```

`FormatStep → TokenizeStep → SplitStep` is a pull-based chain orchestrated through
`.next()`. Dataclasses perform the transformations. A fresh chain of cursors is
created for each encoding call; the processor shares its configured tokenizer.

The full request is formatted into state, query and option messages, then tokenized
once using the exported chat template. Message boundaries let
`TokenizedSample.split()` select state + current query + one option for each
candidate. Other questions and options are excluded from that candidate sequence.

## Candidate scoring

Each backend runs causal attention over a candidate sequence, selects the final
nonpadding token's hidden state, and applies the saved decision head. A linear
head produces one score directly. An MLP reconstructs the saved layer widths,
activation and LayerNorm placement before producing that score.

The Transformers implementation is `SocialJevModel`. The vLLM embedding-model
adapter replaces its default pooler with our trained head: it returns raw scalar
scores, not normalized embeddings or vocabulary-generation scores.
`Processor.decode()` delegates grouping and per-question softmax to `Encoded`.

## Execution modes

Transformers executes a padded batch. Offline vLLM accepts the complete collection
of unpadded token sequences. Online vLLM exposes the same pooling computation
through HTTP. Native caching, chunked prefill and scheduling do not change the
intended classification computation, but precision and kernels can change floating
point results, particularly for close candidate scores.

Examples are currently omitted because their original attention isolation and
logical positions are not implemented by this causal vLLM path. The package does
not provide backward compatibility with its earlier LM-head inference interface.
