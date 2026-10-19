# Concepts

SocialJev expresses a task as typed requests and returns typed results. These concepts are shared by the models' direct Transformers interface and the SocialJev package, including its Transformers and vLLM backends.

## Queries

A `Query` describes what you want to find out about a state. It consists of a question (`text`) and candidate answers (`options`). The question and options together define the task; the model selects among the options you supply.

### Question

The `text` field contains the question the model should answer about the state:

```python
text = "Which policy area does this statement concern?"
```

The question defines the comparison. Asking whether a statement *mentions* public transport is a different task from asking whether it *supports expanding* it, even if the candidate answers are the same.

### Options

The `options` field contains the possible answers to that question:

```python
options = ["Transport", "Education", "Healthcare"]
```

Options are answer texts, rather than fixed numerical label IDs. Each query must contain at least one option, and its options must be distinct. Their order is preserved in the returned probability mapping.

A complete query combines both fields:

```python
from social_jev import Query

policy_query = Query(
    text="Which policy area does this statement concern?",
    options=["Transport", "Education", "Healthcare"],
)
```

## Requests

A `Request` pairs a **state** with the **queries** to answer about it. The state is the text to examine—for example, a document, a paragraph, or a statement. The query supplies the question and candidate answers introduced above.

Using the `policy_query` already defined:

```python
from social_jev import Request

request = Request(
    state="The council will expand the bus network.",
    queries=[policy_query],
)
```

A request can contain multiple queries about the same state. For example, an additional query can ask whether the statement proposes expanding a public service:

```python
expansion_query = Query(
    text="Does this statement propose expanding a public service?",
    options=["Yes", "No"],
)
request = Request(
    state="The council will expand the bus network.",
    queries=[policy_query, expansion_query],
)
```

Each request must contain at least one query, and different requests can contain different numbers of queries.

The same query can also be reused across states:

```python
requests = [
    Request(state="The council expanded the bus network.", queries=[policy_query]),
    Request(state="The government recruited more teachers.", queries=[policy_query]),
]
```

When loading a model directly through Transformers, the corresponding constructors are available as `model.Query` and `model.Request`.

## Results

Each query produces a typed `Result` with two attributes:

- **`choice`**: the supplied option with the highest probability.
- **`probabilities`**: a mapping from each supplied option to its probability.

```python
results = model.query(request)
result = results[0][0]
print(result.choice)
print(result.probabilities)
```

A possible output for the policy-area query is:

```text
Transport
{'Transport': 0.95, 'Education': 0.03, 'Healthcare': 0.02}
```

Results are ordered **first by document, then by query for that document**. Both submission methods return a nested list, including when submitting only one request.

For the request containing both `policy_query` and `expansion_query`:

```python
results = model.query(request)

results[0][0]  # First document: policy-area query.
results[0][1]  # First document: public-service expansion query.
```

For a collection of requests, `results[1][0]` would select the first query about the second document. Each concurrent submission returns results for its own requests with the same ordering.

The model assigns a score to each document–query–option combination. SocialJev converts the scores into probabilities using softmax across the options for that query. A checkpoint can use a trained linear or MLP decision head; the saved checkpoint determines which one is loaded.

Probabilities sum to one within the supplied options. Adding or changing options changes the comparison. The values are not independent probabilities for each label, nor automatically calibrated estimates of real-world certainty.

## Sync and Async inference

The package exposes two submission methods:

- **`.query()`** submits requests synchronously. The calling code waits until the results are returned before continuing.
- **`.aquery()`** submits requests asynchronously. Awaiting its results allows other asynchronous work, including other submissions, to proceed.

Both methods accept one `Request` or a list of requests, and both return the nested results described above. Asynchronous execution does not require a particular batch size or data source; it provides the ability to overlap work.

The default Transformers backend supports `.query()`. Both local vLLM and the managed vLLM server support `.query()` and concurrent `.aquery()` calls.

The [inference guide](inference.md) explains how to choose a submission pattern and configure a workload. The [API reference](api.md) lists the arguments and return types.
