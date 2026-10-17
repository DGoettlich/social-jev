"""Offline vLLM pooling for the fixed scalar classification method."""

import asyncio
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
    """Load a local vLLM model for synchronous or asynchronous inference.

    Args:
        repo: Self-contained scalar checkpoint.

    Examples:
        >>> engine = VllmEngine(repo)  # doctest: +SKIP
        >>> engine.stop()  # doctest: +SKIP
    """

    def __init__(self, repo, **kwargs):
        settings = engine_settings(**kwargs)
        settings["pooler_config"] = PoolerConfig(**settings["pooler_config"])
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
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(model=str(repo), **settings))

    async def submit_query_option_sequence(self, ids: list[int]) -> float:
        assert self.engine is not None
        output = None
        async for result in self.engine.encode(
            {"prompt_token_ids": ids},
            PoolingParams(task="embed"),
            request_id=str(next(self.request_ids)),
        ):
            output = result
        assert output is not None
        assert output.finished
        return output.outputs.data.item()

    async def submit_as_query_option_sequences(self, inputs: list[list[int]]) -> list[float]:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(self.submit_query_option_sequence(ids)) for ids in inputs]
        return [task.result() for task in tasks]

    def process(self, inputs: list[list[int]]) -> list[float]:
        """Process token sequences synchronously and return scalar scores in input order."""
        return asyncio.run_coroutine_threadsafe(
            self.submit_as_query_option_sequences(inputs), self.loop
        ).result()

    async def aprocess(self, inputs: list[list[int]]) -> list[float]:
        """Process token sequences asynchronously and return scalar scores in input order."""
        future = asyncio.run_coroutine_threadsafe(
            self.submit_as_query_option_sequences(inputs), self.loop
        )
        return await asyncio.wrap_future(future)

    async def shutdown(self):
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None
        await self.loop.shutdown_asyncgens()

    def stop(self):
        """Stop the engine, release its workers, and close its event-loop thread."""
        if self.loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self.shutdown(), self.loop).result()
            finally:
                self.loop.call_soon_threadsafe(self.loop.stop)
                self.thread.join()
                self.loop.close()
