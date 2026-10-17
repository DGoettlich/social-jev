"""Exercise the local async adapter with CPU scoring at the native vLLM boundary."""

import asyncio
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from social_jev import Query, Request, SocialJev
from social_jev.transformers import TransformersEngine


def test_local_queries_share_engine_and_keep_results_with_their_call(  # noqa: C901 -- Exercise one complete local engine lifecycle.
    checkpoint, monkeypatch
):
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    reference = SocialJev.from_pretrained(checkpoint)
    items = [
        Request(state="S", queries=[Query(text="Q", options=["A", "B"])]),
        Request(state="S S", queries=[Query(text="R", options=["C", "B", "A"])]),
    ]
    expected = [reference.query(item) for item in items]
    reference.stop()
    seen = set()
    active = set()
    overlap = []
    cancelled = asyncio.Event()

    class CpuAsyncLLM:
        def __init__(self, args):
            self.model = TransformersEngine(args.model)
            self.args = args
            self.hold = False

        @classmethod
        def from_engine_args(cls, args):
            assert "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ
            return cls(args)

        async def encode(self, prompt, params, request_id):
            assert request_id not in seen
            seen.add(request_id)
            active.add(request_id)
            try:
                assert params.task == "embed"
                assert params.use_activation is False
                if self.hold:
                    await asyncio.sleep(60)
                await asyncio.sleep(0.01)
                overlap.append(len(active))
                score = self.model.score([prompt["prompt_token_ids"]])[0]
                yield SimpleNamespace(
                    finished=True, outputs=SimpleNamespace(data=torch.tensor(score))
                )
            except asyncio.CancelledError:
                cancelled.set()
                raise
            finally:
                active.remove(request_id)

        def shutdown(self):
            self.model.stop()

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            AsyncEngineArgs=SimpleNamespace,
            PoolingParams=SimpleNamespace,
        ),
    )
    monkeypatch.setitem(sys.modules, "vllm.config", SimpleNamespace(PoolerConfig=SimpleNamespace))
    monkeypatch.setitem(
        sys.modules, "vllm.v1.engine.async_llm", SimpleNamespace(AsyncLLM=CpuAsyncLLM)
    )
    # Import this adapter against the substituted optional dependency for this test only.
    monkeypatch.delitem(sys.modules, "social_jev.vllm.offline", raising=False)
    model = SocialJev.from_pretrained(checkpoint, backend="vllm", tensor_parallel_size=2)

    async def concurrent():
        actual = await asyncio.gather(
            model.aquery(items[0]),
            model.aquery(items[1]),
            asyncio.to_thread(model.query, items[1]),
        )
        assert actual == [expected[0], expected[1], expected[1]]
        assert await model.aquery([]) == []
        model.engine.engine.hold = True
        task = asyncio.create_task(model.aquery(items[0]))
        while not active:
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        while not cancelled.is_set() or active:
            await asyncio.sleep(0.001)
        model.engine.engine.hold = False
        assert await model.aquery(items[1]) == expected[1]

    try:
        assert model.query(items[0]) == expected[0]
        assert model.query([]) == []
        assert model.engine.engine.args.tensor_parallel_size == 2
        asyncio.run(asyncio.wait_for(concurrent(), timeout=10))
        assert max(overlap) > 1
    finally:
        model.stop()
        model.stop()
        monkeypatch.delitem(sys.modules, "social_jev.vllm.offline", raising=False)
    assert not model.engine.thread.is_alive()
    assert model.engine.loop.is_closed()
