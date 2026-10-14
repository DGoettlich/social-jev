"""Public typed classification through Transformers or vLLM."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from .config import SocialJevConfig
from .data import Processor

if TYPE_CHECKING:
    from .types import Request, Result


class SocialJev:
    """Classify documents using a trained Qwen3 linear-head checkpoint.

    Args:
        processor: Checkpoint request preparation and result decoding.
        engine: Selected execution engine.
        serve: Whether the engine accepts asynchronous server requests.

    Examples:
        >>> model = SocialJev.from_pretrained(repo)  # doctest: +SKIP
        >>> results = model.query([request])  # doctest: +SKIP
        >>> results[0][0].choice  # doctest: +SKIP
        'Transport'
        >>> model.stop()  # doctest: +SKIP
    """

    def __init__(self, processor: Processor, engine, *, serve: bool = False):
        """Attach the processor and execution engine."""
        self.processor = processor
        self.engine = engine
        self.serve = serve

    @classmethod
    def from_pretrained(cls, repo, *, backend="transformers", serve=False):
        """Load a linear checkpoint for offline inference or async vLLM serving."""
        if backend not in {"transformers", "vllm"} or (serve and backend != "vllm"):
            message = "Choose transformers offline, vllm offline, or vllm with serve=True"
            raise ValueError(message)
        config = SocialJevConfig.from_pretrained(repo)
        processor = Processor.from_pretrained(repo, config.max_position_embeddings)
        if backend == "transformers":
            from .transformers import TransformersEngine  # noqa: PLC0415

            engine = TransformersEngine(repo, processor.tokenizer)
        elif serve:
            from .server import Server  # noqa: PLC0415

            engine = Server(repo)
        else:
            from .vllm import VllmEngine  # noqa: PLC0415

            engine = VllmEngine(repo)
        return cls(processor, engine, serve=serve)

    def query(self, requests: Request | list[Request]) -> list[list[Result]]:
        """Classify offline and always return document/question nested results."""
        if self.serve:
            message = "Use await model.aquery() for online vLLM inference"
            raise RuntimeError(message)
        encoded = self.processor.encode(requests)
        scores = self.engine.score(encoded.inputs)
        return self.processor.decode(scores, encoded)

    async def aquery(self, requests: Request | list[Request]) -> list[list[Result]]:
        """Keep CPU preparation and decoding off the caller's event-loop thread."""
        if not self.serve:
            message = "aquery requires backend='vllm', serve=True"
            raise RuntimeError(message)
        encoded = await asyncio.to_thread(self.processor.encode, requests)
        scores = await self.engine.ascore(encoded.inputs)
        return await asyncio.to_thread(self.processor.decode, scores, encoded)

    def stop(self):
        """Release owned resources after outstanding queries have finished."""
        self.engine.stop()
