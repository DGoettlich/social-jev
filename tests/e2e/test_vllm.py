import asyncio
import os

import pytest

from social_jev import Query, Request, SocialJev

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("SOCIAL_JEV_GPU_TESTS") != "1",
        reason="Set SOCIAL_JEV_GPU_TESTS=1 on a Linux CUDA host with social-jev[vllm] installed",
    ),
    pytest.mark.filterwarnings("ignore:.*torch.jit.*is deprecated:DeprecationWarning"),
    pytest.mark.filterwarnings("ignore:Unclosed context:ResourceWarning"),
]


def test_native_vllm_checkpoint_loading_offline_and_online(checkpoint):
    request = Request(state=" ".join(["S"] * 80), queries=[Query(text="Q", options=["A", "B"])])
    reference = SocialJev.from_pretrained(checkpoint)
    items = [
        request,
        Request(state="S", queries=[Query(text="R", options=["B", "C", "A"])]),
    ]
    expected_batch = reference.query(items)
    reference.stop()

    async def check_calls(model):
        batches = [items, list(reversed(items)), items[:1]]
        expected_batches = [expected_batch, list(reversed(expected_batch)), expected_batch[:1]]
        results = await asyncio.gather(
            *(model.aquery(batch) for batch in batches),
            asyncio.to_thread(model.query, items),
        )
        for actual, expected_groups in zip(
            results, [*expected_batches, expected_batch], strict=True
        ):
            assert [len(group) for group in actual] == [len(group) for group in expected_groups]
            for group, expected_group in zip(actual, expected_groups, strict=True):
                for result, wanted in zip(group, expected_group, strict=True):
                    assert result.probabilities == pytest.approx(wanted.probabilities, abs=0.01)
        assert model.query([]) == []
        assert await model.aquery([]) == []

    for serve in (False, True):
        model = SocialJev.from_pretrained(
            checkpoint, backend="vllm", serve=serve, max_num_batched_tokens=16, max_num_seqs=1
        )
        try:
            actual = model.query(request)
            assert actual[0][0].probabilities == pytest.approx(
                expected_batch[0][0].probabilities, abs=0.01
            )
            asyncio.run(check_calls(model))
        finally:
            model.stop()
        assert not model.engine.thread.is_alive()
        model.stop()
