# Classify text with SocialJev

SocialJev provides Jev-like classification foundation models for the social sciences,
inspired by TypeSafe Jev. It combines the flexibility of language models trained on
trillions of tokens with the goal of fast, lightweight text classification.
Define your task through a simple typed interface and SocialJev returns typed outputs.

!!! note "Active development"
    SocialJev is an alpha package. Interfaces and checkpoint formats may change
    between releases. These pages describe the version selected in the sidebar.

## Install

```bash
uv pip install social-jev
```

## Load a model and request a batch

```python
from social_jev import Query, Request, SocialJev

model = SocialJev.from_pretrained("social-jev/social-jev-0.6B-preview")
query = Query(
    text="Which policy area does this document concern?",
    options=["Transport", "Education", "Healthcare"],
)
try:
    results = model.query(
        [
            Request(state="The council expanded the bus network.", queries=[query]),
            Request(state="The government recruited more teachers.", queries=[query]),
        ]
    )
    print(results[0][0].model_dump())
finally:
    model.stop()
```

An illustrative result:

```python
{"choice": "Transport", "probabilities": {"Transport": 0.98, "Education": 0.01, "Healthcare": 0.01}}
```

The outer list follows document order; each inner list follows that document's
question order. Probabilities are normalized over the options you supplied.
The numbers above illustrate the output format and are not a guaranteed prediction.

## Choose how to run

Use [offline inference](batch-inference.md) when the full input collection is ready.
Use [async serving](serving.md) when requests arrive continuously or preparation
should overlap GPU execution. Both use the same request and result types.

The preview checkpoint currently has a **linear decision head**. The package also
loads MLP heads when their architecture is saved in the checkpoint.
