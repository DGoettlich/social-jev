# SocialJev

SocialJev is a Jev-like classification foundation model for the social sciences,
inspired by TypeSafe Jev. It is trained to combine the flexibility of modern
language models pretrained on trillions of tokens of text with the speed of
traditional lightweight NLP methods. Define your task through a simple typed
interface and SocialJev returns typed outputs.

## Getting started

```bash
uv pip install social-jev
```

Load a model from Hugging Face and classify a batch of documents:

```python
from social_jev import Query, Request, SocialJev

model = SocialJev.from_pretrained("social-jev/<model-name>")

query = Query(
    text="Which policy area does this document concern?",
    options=["Transport", "Education", "Healthcare"],
)

results = model.query(
    [
        Request(state="The council expanded the bus network.", queries=[query]),
        Request(state="The government recruited more teachers.", queries=[query]),
    ],
    batch_size=16,
)

print(results[0][0].model_dump())
```

Example result:

```python
{
    "choice": "Transport",
    "probabilities": {
        "Transport": 0.91,
        "Education": 0.03,
        "Healthcare": 0.06,
    },
}
```

Results follow document order, with one result per query. The probabilities above
are illustrative.
