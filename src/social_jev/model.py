"""Public typed classification through Transformers or vLLM."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

from transformers import AutoTokenizer

from . import config  # noqa: F401 -- Register the checkpoint configuration before loading.
from .data import Processor

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from .types import Request, Result


class SocialJev:
    """Classify documents using a trained Qwen3 linear or MLP checkpoint.

    Args:
        processor: Checkpoint request preparation and result decoding.
        engine: Selected execution engine.
        serve: Whether vLLM runs behind an owned HTTP server.

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
    def from_pretrained(cls, repo, *, backend="transformers", serve=False, **kwargs):
        """Load a scalar-head checkpoint for offline inference or vLLM serving."""
        if backend not in {"transformers", "vllm"} or (serve and backend != "vllm"):
            message = "Choose transformers offline, vllm offline, or vllm with serve=True"
            raise ValueError(message)
        if backend == "transformers":
            from .transformers import TransformersEngine  # noqa: PLC0415

            engine = TransformersEngine(repo, **kwargs)
            tokenizer = engine.tokenizer
        else:
            tokenizer = AutoTokenizer.from_pretrained(repo, revision=kwargs.get("revision"))
            if serve:
                from .vllm.server import Server  # noqa: PLC0415

                engine = Server(repo, **kwargs)
            else:
                from .vllm.offline import VllmEngine  # noqa: PLC0415

                engine = VllmEngine(repo, **kwargs)
        processor = Processor(cast("PreTrainedTokenizerBase", tokenizer))
        return cls(processor, engine, serve=serve)

    def query(self, requests: Request | list[Request]) -> list[list[Result]]:
        """Classify synchronously and always return document/question nested results."""
        encoded = self.processor.encode(requests)
        scores = self.engine.score(encoded.inputs)
        return self.processor.decode(scores, encoded)

    async def aquery(self, requests: Request | list[Request]) -> list[list[Result]]:
        """Keep CPU preparation and decoding off the caller's event-loop thread."""
        if not hasattr(self.engine, "ascore"):
            message = "aquery requires backend='vllm'"
            raise RuntimeError(message)
        encoded = await asyncio.to_thread(self.processor.encode, requests)
        scores = await self.engine.ascore(encoded.inputs)
        return await asyncio.to_thread(self.processor.decode, scores, encoded)

    def stop(self):
        """Release owned resources after outstanding queries have finished."""
        self.engine.stop()
