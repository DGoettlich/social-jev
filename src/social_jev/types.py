"""Typed document coding requests and results."""

from pydantic import BaseModel, Field, model_validator


class Example(BaseModel):
    state: str
    choice: str


class Query(BaseModel):
    text: str
    options: list[str] = Field(min_length=1)
    examples: list[Example] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_options(self):
        assert len(self.options) == len(set(self.options))
        assert all(example.choice in self.options for example in self.examples)
        return self


class Request(BaseModel):
    state: str
    queries: list[Query] = Field(min_length=1)


class Result(BaseModel):
    choice: str
    probabilities: dict[str, float]
