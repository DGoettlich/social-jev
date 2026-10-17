import asyncio
import json
import os
import subprocess
import sys

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


def test_native_vllm_checkpoint_loading_offline_and_online(  # noqa: C901 -- Exercise both native backend lifecycles.
    checkpoint,
):
    request = Request(state=" ".join(["S"] * 80), queries=[Query(text="Q", options=["A", "B"])])
    items = [
        request,
        Request(state="S", queries=[Query(text="R", options=["B", "C", "A"])]),
    ]
    # Keep the CPU reference's CUDA graph-capture checks out of the vLLM parent process.
    reference = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
import sys
from social_jev import Request, SocialJev
model = SocialJev.from_pretrained(sys.argv[1])
items = [Request.model_validate(item) for item in json.load(sys.stdin)]
results = model.query(items)
print(json.dumps([[result.probabilities for result in group] for group in results]))
model.stop()
""",
            str(checkpoint),
        ],
        input=json.dumps([item.model_dump() for item in items]),
        capture_output=True,
        text=True,
        check=True,
    )
    expected_batch = json.loads(reference.stdout)

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
                    assert result.probabilities == pytest.approx(wanted, abs=0.01)
        assert model.query([]) == []
        assert await model.aquery([]) == []

    async def check_backend(serve):
        model = SocialJev.from_pretrained(
            checkpoint, backend="vllm", serve=serve, max_num_batched_tokens=16, max_num_seqs=1
        )
        try:
            actual = model.query(request)
            assert actual[0][0].probabilities == pytest.approx(expected_batch[0][0], abs=0.01)
            await check_calls(model)
        finally:
            model.stop()
        assert not model.engine.thread.is_alive()
        model.stop()

    for serve in (False, True):
        asyncio.run(check_backend(serve))
