"""Offline vLLM pooling for the fixed scalar classification method."""

from vllm import (  # ty: ignore[unresolved-import] -- Optional Linux GPU dependency.
    LLM,
    PoolingParams,
)
from vllm.config import PoolerConfig  # ty: ignore[unresolved-import]

from . import engine_settings


class VllmEngine:
    """Submit all candidate sequences once and let vLLM schedule execution.

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
        self.engine = LLM(model=str(repo), **settings)

    def score(self, inputs: list[list[int]]) -> list[float]:
        """Return one raw scalar score per unpadded candidate."""
        if not inputs:
            return []
        assert self.engine is not None
        outputs = self.engine.encode(
            [{"prompt_token_ids": ids} for ids in inputs],
            pooling_task="embed",
            pooling_params=PoolingParams(use_activation=False),
            use_tqdm=False,
        )
        return [output.outputs.data.item() for output in outputs]

    def stop(self):
        """Shut down the owned engine and release its worker processes."""
        if self.engine is not None:
            self.engine.llm_engine.engine_core.shutdown()
            self.engine = None
