# Requests and results

A **state** is the document to classify. A **query** describes the task and its
candidate answers. One request contains one state and one or more queries.

## Define the task

```python
from social_jev import Query, Request

request = Request(
    state="The council will add electric buses and expand local rail services.",
    queries=[
        Query(text="Policy area?", options=["Transport", "Education"]),
        Query(text="Does this propose an expansion?", options=["Yes", "No"]),
    ],
)
```

Candidate answers are strings in a nonempty options list, with no duplicates.
Use meaningful descriptions rather than requiring a fixed set of trained label
IDs. An example-based prompt interface is not currently supported.

## Read typed outputs

Both `model.query(request)` and `model.query([request])` return
`list[list[Result]]`. For this request, `results[0]` contains two results.

```python
first = results[0][0]
print(first.choice)
print(first.probabilities)
print(first.model_dump())
```

Each `Result` contains the highest-scoring candidate in `choice` and an ordered
mapping from supplied candidate text to probability in `probabilities`.
Documents, questions and candidate keys retain their input order.

## Interpret the probabilities

The head assigns one raw score to each candidate. SocialJev applies softmax within
each question, so its candidate probabilities sum to one. Adding or changing
candidate answers changes that comparison. These values are not automatically
calibrated measures of real-world certainty, nor independent multi-label scores.

An empty collection returns `[]`. An individual request must contain at least one
query, and each query must contain at least one distinct option.
