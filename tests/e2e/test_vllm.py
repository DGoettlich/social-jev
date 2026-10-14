import asyncio
import os

import pytest

from social_jev import Query, Request, SocialJev

pytestmark = pytest.mark.skipif(
    os.environ.get("SOCIAL_JEV_GPU_TESTS") != "1",
    reason="Set SOCIAL_JEV_GPU_TESTS=1 on a Linux CUDA host with social-jev[vllm] installed",
)


def test_native_vllm_checkpoint_loading_offline_and_online(checkpoint):
    request = Request(state="S", queries=[Query(text="Q", options=["A", "B"])])
    reference = SocialJev.from_pretrained(checkpoint)
    expected = reference.query(request)
    reference.stop()
    model = SocialJev.from_pretrained(checkpoint, backend="vllm")
    try:
        actual = model.query(request)
        assert actual[0][0].probabilities == pytest.approx(expected[0][0].probabilities, abs=0.01)
    finally:
        model.stop()
    online = SocialJev.from_pretrained(checkpoint, backend="vllm", serve=True)
    try:
        actual = asyncio.run(online.aquery(request))
        assert actual[0][0].probabilities == pytest.approx(expected[0][0].probabilities, abs=0.01)
    finally:
        online.stop()
