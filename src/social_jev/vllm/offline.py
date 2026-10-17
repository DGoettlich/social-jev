"""Offline vLLM pooling for the fixed scalar classification method."""

import asyncio
import os
import threading
from itertools import count

from vllm import (  # ty: ignore[unresolved-import] -- Optional Linux GPU dependency.
    AsyncEngineArgs,
    PoolingParams,
)
from vllm.config import PoolerConfig  # ty: ignore[unresolved-import]
from vllm.v1.engine.async_llm import AsyncLLM  # ty: ignore[unresolved-import]

from . import engine_settings


class VllmEngine:
    """Share a native asynchronous vLLM engine across synchronous and async calls.

    In standalone scripts, construct the engine inside an
    ``if __name__ == "__main__":`` guard so spawned workers can import the script.

    Args:
        repo: Self-contained scalar checkpoint.

    Examples:
        >>> engine = VllmEngine(repo)  # doctest: +SKIP
        >>> engine.stop()  # doctest: +SKIP
    """

    def __init__(self, repo, **kwargs):
        """Load the native pooling model with prefix caching."""
        settings = engine_settings(**kwargs)
        settings["pooler_config"] = PoolerConfig(**settings["pooler_config"])
        # Avoid inheriting CUDA driver state left by earlier work in the caller's process.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        self.request_ids = count()
        self.engine = None
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        try:
            asyncio.run_coroutine_threadsafe(self.start(repo, settings), self.loop).result()
        except BaseException:
            self.stop()
            raise

    async def start(self, repo, settings):
        """Construct the engine on the event loop that owns its output handler."""
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(model=str(repo), **settings))

    async def candidate(self, ids: list[int]) -> float:
        """Collect one candidate's final score through its own request stream."""
        assert self.engine is not None
        output = None
        async for result in self.engine.encode(
            {"prompt_token_ids": ids},
            PoolingParams(task="embed", use_activation=False),
            request_id=str(next(self.request_ids)),
        ):
            output = result
        assert output is not None
        assert output.finished
        return output.outputs.data.item()

    async def request(self, inputs: list[list[int]]) -> list[float]:
        """Run candidates concurrently and retain their input order."""
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(self.candidate(ids)) for ids in inputs]
        return [task.result() for task in tasks]

    def score(self, inputs: list[list[int]]) -> list[float]:
        """Wait for ordered scalar scores on the engine's event loop."""
        return asyncio.run_coroutine_threadsafe(self.request(inputs), self.loop).result()

    async def ascore(self, inputs: list[list[int]]) -> list[float]:
        """Await ordered scalar scores without blocking the caller's event loop."""
        future = asyncio.run_coroutine_threadsafe(self.request(inputs), self.loop)
        return await asyncio.wrap_future(future)

    async def shutdown(self):
        """Release workers and finish the engine's asynchronous generators."""
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None
        await self.loop.shutdown_asyncgens()

    def stop(self):
        """Shut down the owned engine and its event-loop thread."""
        if self.loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self.shutdown(), self.loop).result()
            finally:
                self.loop.call_soon_threadsafe(self.loop.stop)
                self.thread.join()
                self.loop.close()
