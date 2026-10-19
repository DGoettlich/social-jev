# SocialJev

SocialJev is a Jev-like classification foundation model for the social sciences,
inspired by TypeSafe Jev. It is trained to combine the flexibility of modern
language models pretrained on trillions of tokens of text with the speed of
traditional lightweight NLP methods. Define your task through a simple typed
interface and SocialJev returns typed outputs.

**Note**
SocialJev is under active development. The API and model interfaces may change,
including breaking changes between releases.

## Getting started

```bash
uv pip install social-jev
```

Load a model from Hugging Face and classify a batch of documents:

```python
from social_jev import Query, Request, SocialJev

model = SocialJev.from_pretrained("social-jev/social-jev-0.6B-preview")

query = Query(
    text="Which policy area does this document concern?",
    options=["Transport", "Education", "Healthcare"],
)

results = model.query(
    [
        Request(state="The council expanded the bus network.", queries=[query]),
        Request(state="The government recruited more teachers.", queries=[query]),
    ],
)

print(results[0][0].choice)
print(results[0][0].probabilities)
```

Example result:

```text
Transport
{'Transport': 0.91, 'Education': 0.03, 'Healthcare': 0.06}
```

Results are ordered first by document, then by query for that document. The
probabilities above are illustrative.
