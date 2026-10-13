# Offline LM-head inference

Install with `pip install social-jev` once the package is released. For development,
run `uv sync` in the repository checkout. Python 3.12 or later is required.

```python
from social_jev import Query, Request, SocialJev

model = SocialJev.from_pretrained("/path/to/exported-checkpoint", local_files_only=True)
requests = [
    Request(
        state="The council increased public transport funding.",
        queries=[Query(text="Which policy area is discussed?", options=["Transport", "Education"])],
    ),
]
results = model.query(requests, batch_size=16)
print(results[0][0].choice)
print(results[0][0].probabilities)
```

A single `Request` returns `list[Result]`, one result per query. A list of requests
returns `list[list[Result]]`, preserving document and query order. `batch_size`
counts documents. An empty request list returns an empty result list.

`model.processor.encode(requests)` prepares one CPU batch. For multiple batches,
`model.processor.batches(requests, batch_size=16)` returns a pull-based chain;
call `.next()` until it returns `None`. Internally both interfaces use
`RequestStep → FormatStep → TokenizeStep → PadStep → CollateStep`.

Dataclasses own the transformations: `Sample.from_request()` formats messages,
`TokenizedSample.from_sample()` tokenizes them, `TokenizedSample.pad()` pads tokens,
and `Batch.from_samples()` builds CPU tensors. Steps only pull inputs, arrange
batches, and call these methods; `Processor` assembles the chain.

Load a complete SocialJev LM-head HF export, including its tokenizer and saved
chat template. Other decision heads and plain causal-LM checkpoints are not
supported. The saved template remains authoritative; `Processor.template()`
provides the bundled LM-head template for preparing compatible tokenizer assets.

Hugging Face identifiers can download model assets initially. Local exports run
without network access. Tokenizer assets must be present alongside the weights.
Requests exceeding context capacity raise an error; documents are not truncated
or split. Each option code must tokenize to one distinct known answer token.

Probabilities are normalized over the options supplied for that query. They do
not establish calibration or accuracy for a social science task. This package
contains inference only: training, evaluation pipelines, serving, and other heads
remain outside its scope.
