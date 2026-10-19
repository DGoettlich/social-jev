# Getting started

SocialJev is a family of 'classification foundation models' for the social sciences, inspired by [TypeSafe's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev). The models combine the flexibility of modern language models trained on trillions of tokens of text with the speed of lightweight NLP methods. Define your task through a simple typed interface and SocialJev returns typed outputs.

The models can be used **directly through Transformers**, without installing SocialJev, or **through the SocialJev package**, which also provides optional vLLM integrations for larger workloads.

> **Active development** — SocialJev is under active development. Interfaces may change between early releases.

## Quickstart (Transformers)

A checkpoint can be loaded from Huggingface and queried directly without installing the SocialJev package:

```bash
uv pip install torch transformers pydantic
```

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "social-jev/social-jev-0.6B-preview",
    trust_remote_code=True,
)
request = model.Request(
    state="The council expanded the bus network.",
    queries=[
        model.Query(
            text="Which policy area does this document concern?",
            options=["Transport", "Education", "Healthcare"],
        ),
    ],
)

results = model.query(request)
print(results[0][0].choice)
print(results[0][0].probabilities)
```

A possible output might be:

```text
Transport
{'Transport': 0.98, 'Education': 0.01, 'Healthcare': 0.01}
```

## Sustained workloads with the SocialJev package

The package provides the same typed interface across inference backends. For larger sustained workloads, its optional vLLM integration is optimized for fast inference.

The standard PyPI installation uses Transformers:

```bash
uv pip install social-jev
```

To include the vLLM dependencies:

```bash
uv pip install "social-jev[vllm]"
```

Then load the model, define a query, and submit it to the model. The package provides two methods for submitting queries:

1. **.query()** submits queries synchronously
2. **.aquery()** submits them asynchronously

We provide more details on the differences between the methods and different execution backends on the [inference page](inference.md). The example below uses vLLM with the synchronous method, which you can use like so:

```python
from social_jev import Query, Request, SocialJev

model = SocialJev.from_pretrained(
    "social-jev/social-jev-0.6B-preview",
    backend="vllm",
)
request = Request(
    state="The council expanded the bus network.",
    queries=[
        Query(
            text="Which policy area does this document concern?",
            options=["Transport", "Education", "Healthcare"],
        ),
    ],
)

try:
    results = model.query(request)
    print(results[0][0].choice)
    print(results[0][0].probabilities)
finally:
    model.stop()
```

The returned results have the same structure as in the Transformers example above.

## Roadmap

This is an early release. In particular, we are in the process of:

1. Scaling the models and training pipeline; if these models do not yet seem to perform well on your task, consider shooting us an [email](mailto:dgoettlich@ethz.ch) and bear with us: future models will be trained on more diverse datasets and queries.
2. Adding support for various GPU-providers so you can launch inference jobs without handling the GPU rental and setup yourself
3. Adding a simple training interface so you can fine-tune your own model
4. Adding an evaluation interface so you can evaluate the performance of SocialJev models under various queries easily
5. Adding an interface to optimize SocialJev queries for your task
