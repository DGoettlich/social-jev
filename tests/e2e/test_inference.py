import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from social_jev import Query, Request, SocialJev


def requests():
    return [
        Request(
            state="S",
            queries=[Query(text="Q", options=["A", "B"]), Query(text="R", options=["C", "A", "B"])],
        ),
        Request(state="S S", queries=[Query(text="R", options=["B", "C"])]),
    ]


def test_checkpoint_classifies_and_preserves_nested_order(checkpoint):
    model = SocialJev.from_pretrained(checkpoint)
    items = requests()
    actual = model.query(items)
    saved_config = json.loads((checkpoint / "config.json").read_text())
    assert saved_config["model_type"] == "social_jev"
    assert saved_config["backbone_config"]["model_type"] == "qwen3"
    assert saved_config["backbone_config"]["hidden_size"] == 128
    assert "hidden_size" not in saved_config
    assert saved_config["decision_head"]["type"] in {"linear", "mlp"}
    assert [len(group) for group in actual] == [2, 1]
    for item, group in zip(items, actual, strict=True):
        single = model.query(item)
        assert len(single) == 1
        for query, result, one in zip(item.queries, group, single[0], strict=True):
            assert list(result.probabilities) == query.options
            assert result.probabilities == pytest.approx(one.probabilities, abs=1e-6)
            assert sum(result.probabilities.values()) == pytest.approx(1)
            assert result.choice == max(result.probabilities, key=result.probabilities.get)
    assert model.query([]) == []
    reloaded = SocialJev.from_pretrained(checkpoint)
    assert reloaded.query(items) == actual
    model.stop()
    model.stop()
    reloaded.stop()


def test_candidates_are_isolated_and_processor_is_reentrant(checkpoint):
    model = SocialJev.from_pretrained(checkpoint)
    item = requests()[0]
    combined = model.query(item)[0]
    for query, expected in zip(item.queries, combined, strict=True):
        actual = model.query(Request(state=item.state, queries=[query]))[0][0]
        assert actual.probabilities == pytest.approx(expected.probabilities, abs=1e-6)
    processor = model.processor
    with ThreadPoolExecutor(max_workers=2) as executor:
        encoded = list(executor.map(processor.encode, requests()))
    assert [len(sample.inputs) for sample in encoded] == [5, 2]
    for sample in encoded:
        for ids in sample.inputs:
            text = processor.tokenizer.decode(ids, skip_special_tokens=False)
            assert text.count("state") == 1
            assert text.count("query") == 1
            assert text.count("option") == 1
            assert ids[-1] == processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert processor.decode([], processor.encode([])) == []
    model.stop()


def test_context_limit_and_execution_modes_fail_clearly(checkpoint):
    model = SocialJev.from_pretrained(checkpoint)
    with pytest.raises(ValueError, match="capacity"):
        model.query(Request(state="S " * 300, queries=requests()[0].queries))
    with pytest.raises(RuntimeError, match="aquery requires"):
        asyncio.run(model.aquery(requests()))
    with pytest.raises(ValueError, match="Choose"):
        SocialJev.from_pretrained(checkpoint, serve=True)
    model.stop()


def test_format_tokenize_split_runs_once_per_document(checkpoint, monkeypatch):
    model = SocialJev.from_pretrained(checkpoint)
    formatted = []
    apply_template = model.processor.tokenizer.apply_chat_template

    def tokenize(messages, **kwargs):
        formatted.append(messages)
        return apply_template(messages, **kwargs)

    monkeypatch.setattr(model.processor.tokenizer, "apply_chat_template", tokenize)
    items = requests()
    try:
        results = model.query(items)
        assert [len(group) for group in results] == [2, 1]
        assert len(formatted) == 2
        assert [message["role"] for message in formatted[0]] == [
            "state",
            "query",
            "option",
            "option",
            "query",
            "option",
            "option",
            "option",
        ]
        assert [message["content"] for message in formatted[0]] == [
            "S",
            "Q",
            "A",
            "B",
            "R",
            "C",
            "A",
            "B",
        ]
    finally:
        model.stop()


def test_transformers_loading_kwargs_reach_model(checkpoint):
    model = SocialJev.from_pretrained(checkpoint, dtype=torch.float64, local_files_only=True)
    try:
        assert model.engine.model.dtype == torch.float64
        assert model.processor.tokenizer is not None
        assert len(model.query(requests())) == 2
    finally:
        model.stop()
