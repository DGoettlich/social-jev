import json
import shutil
from pathlib import Path

import pytest
import torch
from transformers import AutoModel

from social_jev import Example, Query, Request, SocialJev


def requests():
    return [
        Request(
            state="S",
            queries=[
                Query(text="Q", options=["A", "B"], examples=[Example(state="S", choice="A")]),
                Query(text="R", options=["C", "A", "B"]),
            ],
        ),
        Request(state="S S", queries=[Query(text="R", options=["B", "C"])]),
    ]


def test_batched_results_match_single_and_reload(model, tmp_path):
    items = requests()
    expected = [model.query(item) for item in items]
    actual = model.query(items, batch_size=2)
    remote = AutoModel.from_pretrained(tmp_path, trust_remote_code=True)
    for left_group, right_group, remote_group in zip(
        expected, actual, remote.query(items), strict=True
    ):
        for left, right, reloaded in zip(left_group, right_group, remote_group, strict=True):
            assert left.probabilities == pytest.approx(right.probabilities, abs=1e-6)
            assert left.probabilities == pytest.approx(reloaded.probabilities, abs=1e-6)
            assert sum(right.probabilities.values()) == pytest.approx(1)
            assert right.choice == max(right.probabilities, key=right.probabilities.get)
    assert model.query([]) == []
    assert isinstance(model, SocialJev)


def test_processor_chain_is_fresh_and_preserves_boundaries(model):
    processor = model.processor
    batch = processor.encode(requests())
    chain = processor.batches(requests(), batch_size=1)
    first = chain.next()
    second = chain.next()
    assert first is not None
    assert second is not None
    assert chain.next() is None
    torch.testing.assert_close(first.input_ids[0], batch.input_ids[0])
    assert batch.roles[0] == [
        "state",
        "query",
        "example",
        "option",
        "option",
        "query",
        "option",
        "option",
        "option",
    ]
    decoded = processor.tokenizer.decode(batch.input_ids[0], skip_special_tokens=False)
    assert "A:" in decoded
    assert "answer" in decoded
    assert processor.encode(requests()[0]).roles == first.roles


def test_errors_are_explicit(model):
    with pytest.raises(ValueError, match="positive"):
        model.query(requests(), batch_size=0)
    with pytest.raises(ValueError, match="at least one"):
        model.processor.encode([])
    with pytest.raises(ValueError, match="option codes"):
        model.query(Request(state="S", queries=[Query(text="Q", options=["A", "B", "C", "D"])]))
    with pytest.raises(ValueError, match="capacity"):
        model.query(Request(state="S " * 300, queries=requests()[0].queries))


def test_query_isolation_and_candidate_positions(model):
    item = requests()[0]
    combined = model.query(item)
    for query, expected in zip(item.queries, combined, strict=True):
        actual = model.query(Request(state=item.state, queries=[query]))[0]
        assert actual.probabilities == pytest.approx(expected.probabilities, abs=1e-6)


def test_missing_answer_prefix_rejected(model):
    model.processor.tokenizer.chat_template = (
        "{{ role_start + 'state\\nS' + role_end + role_start + 'query\\nQ' "
        "+ role_end + role_start + 'option\\nA' + role_end }}"
    )
    with pytest.raises(ValueError, match="answer prefix"):
        model.query(Request(state="S", queries=[Query(text="Q", options=["A"])]))


def test_cached_hf_loading_and_loading_info(model, tmp_path):

    cache = tmp_path / "cache"
    repository = cache / "models--social-jev--fixture"
    snapshot = repository / "snapshots" / "fixture"
    snapshot.parent.mkdir(parents=True)
    shutil.copytree(model.name_or_path, snapshot, ignore=shutil.ignore_patterns("cache"))
    (repository / "refs").mkdir()
    (repository / "refs" / "main").write_text("fixture")
    loaded, info = SocialJev.from_pretrained(
        "social-jev/fixture",
        cache_dir=cache,
        local_files_only=True,
        output_loading_info=True,
    )
    assert not info["missing_keys"]
    assert not info["unexpected_keys"]
    assert loaded.query(requests()) == model.query(requests())


def test_predictions_match_mert_reference(model):
    # Generated with MERT's LM-head implementation and the deterministic CPU fixture.
    reference = json.loads(Path("tests/e2e/fixtures/mert_lm_head.json").read_text())
    actual = model.query(requests(), batch_size=2)
    for group, expected_group in zip(actual, reference, strict=True):
        for result, expected in zip(group, expected_group, strict=True):
            assert result.choice == expected["choice"]
            assert result.probabilities == pytest.approx(expected["probabilities"], abs=1e-6)
