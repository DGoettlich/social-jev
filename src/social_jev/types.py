"""Typed document coding requests and results."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Query(BaseModel):
    """A classification question with candidate answers.

    Args:
        text: Question describing the classification task.
        options: Nonempty list of distinct candidate answers.

    Examples:
        >>> query = Query(text="Policy area?", options=["Transport", "Education"])
        >>> query.options[0]
        'Transport'
    """

    model_config = ConfigDict(extra="forbid")
    """Reject unsupported query fields instead of silently ignoring them."""

    text: str
    """Question defining the classification task."""
    options: list[str] = Field(min_length=1)
    """Distinct candidate answers in their supplied order."""

    @model_validator(mode="after")
    def validate_options(self):
        """Require distinct candidate options."""
        assert len(self.options) == len(set(self.options))
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
