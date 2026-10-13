"""Typed document coding requests and results."""

from pydantic import BaseModel, Field, model_validator


class Example(BaseModel):
    """A labelled document illustrating a candidate answer.

    Args:
        state: Example document text.
        choice: Answer matching one of the query's options.

    Examples:
        >>> example = Example(state="The bus network expanded.", choice="Transport")
        >>> example.choice
        'Transport'
    """

    state: str
    """Text of the labelled example document."""
    choice: str
    """Candidate answer assigned to the example."""


class Query(BaseModel):
    """A classification question with candidate answers and optional examples.

    Args:
        text: Question describing the classification task.
        options: Nonempty list of distinct candidate answers.
        examples: Labelled documents whose choices appear in options.

    Examples:
        >>> query = Query(text="Policy area?", options=["Transport", "Education"])
        >>> query.options[0]
        'Transport'
    """

    text: str
    """Question defining the classification task."""
    options: list[str] = Field(min_length=1)
    """Distinct candidate answers in their supplied order."""
    examples: list[Example] = Field(default_factory=list)
    """Optional labelled examples for this query."""

    @model_validator(mode="after")
    def validate_options(self):
        """Require distinct options and example choices drawn from those options."""
        assert len(self.options) == len(set(self.options))
        assert all(example.choice in self.options for example in self.examples)
        return self


class Request(BaseModel):
    """A document and the classification questions to ask about it.

    Args:
        state: Document text to classify.
        queries: Nonempty list of questions about the document.

    Examples:
        >>> query = Query(text="Policy area?", options=["Transport", "Education"])
        >>> request = Request(state="The bus network expanded.", queries=[query])
        >>> len(request.queries)
        1
    """

    state: str
    """Document text to classify."""
    queries: list[Query] = Field(min_length=1)
    """Questions to answer about the document, in output order."""


class Result(BaseModel):
    """The selected answer and probabilities over a query's candidate answers.

    Args:
        choice: Candidate answer with the highest probability.
        probabilities: Candidate answers mapped to their normalized probabilities.

    Examples:
        >>> result = Result(choice="Transport", probabilities={"Transport": 0.9, "Education": 0.1})
        >>> result.choice
        'Transport'
    """

    choice: str
    """Candidate answer with the highest probability."""
    probabilities: dict[str, float]
    """Probabilities normalized over the supplied candidate answers."""
