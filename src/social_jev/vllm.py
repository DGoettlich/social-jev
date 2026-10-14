"""Offline vLLM pooling for the fixed linear classification method."""

from vllm import (  # ty: ignore[unresolved-import] -- Optional Linux GPU dependency.
    LLM,
    PoolingParams,
)
from vllm.config import PoolerConfig  # ty: ignore[unresolved-import]


class VllmEngine:
    """Submit all candidate sequences once and let vLLM schedule execution.

    Args:
        repo: Self-contained linear checkpoint.

    Examples:
        >>> engine = VllmEngine(repo)  # doctest: +SKIP
        >>> engine.stop()  # doctest: +SKIP
    """

    def __init__(self, repo):
        """Load the native pooling model with prefix caching."""
        self.engine = LLM(
            model=str(repo),
            runner="pooling",
            hf_overrides={"architectures": ["SocialJevPoolingModel"], "is_causal": True},
            pooler_config=PoolerConfig(seq_pooling_type="LAST", use_activation=False),
            enable_prefix_caching=True,
            enable_chunked_prefill=False,
            gpu_memory_utilization=0.8,
        )

    def score(self, inputs: list[list[int]]) -> list[float]:
        """Return one raw linear score per unpadded candidate."""
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
