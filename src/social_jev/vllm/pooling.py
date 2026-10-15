"""Native vLLM Qwen3 execution with a replicated trained scalar pooler."""

from vllm.model_executor.layers.pooler import (  # ty: ignore[unresolved-import]
    Pooler,
)
from vllm.model_executor.model_loader.weight_utils import (  # ty: ignore[unresolved-import]
    default_weight_loader,
)
from vllm.model_executor.models.adapters import (  # ty: ignore[unresolved-import]
    as_embedding_model,
)
from vllm.model_executor.models.qwen3 import (  # ty: ignore[unresolved-import]
    Qwen3ForCausalLM,
)

from ..head import DecisionHeadConfig, ScalarHead


class ScalarPooler(Pooler):
    """Apply the trained linear or MLP head to completed candidates' terminal tokens.

    Args:
        config: Checkpoint backbone and trained head settings.

    Examples:
        >>> pooler = ScalarPooler(config)  # doctest: +SKIP
    """

    def __init__(self, config):
        """Create the replicated scalar head on the worker's allocation device."""
        super().__init__()
        self.decision_head = ScalarHead(
            DecisionHeadConfig(**config.decision_head), config.hidden_size
        )

    def get_supported_tasks(self):
        """Expose scalar logits through vLLM's pooling transport."""
        return {"embed"}

    def forward(self, hidden_states, pooling_metadata):
        """Select the final token and return raw logits without normalization."""
        cursor = pooling_metadata.get_pooling_cursor()
        hidden = hidden_states[cursor.last_token_indices_gpu]
        dtype = next(self.decision_head.parameters()).dtype
        logits = self.decision_head(hidden.to(dtype)).float().unsqueeze(-1)
        return list(logits.unbind())


class PoolingModel(as_embedding_model(Qwen3ForCausalLM)):
    """Use vLLM's native Qwen3 backbone and the checkpoint's linear or MLP head.

    Args:
        vllm_config: Engine-supplied model configuration.
        prefix: Engine-supplied parameter prefix.

    Examples:
        >>> from social_jev.vllm.plugin import register
        >>> register()  # doctest: +SKIP
    """

    def __init__(self, *, vllm_config, prefix=""):
        """Replace the embedding wrapper's default pooler with scalar scoring."""
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.pooler = ScalarPooler(vllm_config.model_config.hf_config)

    def load_weights(self, weights):
        """Route backbone weights to native loading and head weights to the pooler."""
        loaded_head = set()
        parameters = dict(self.pooler.decision_head.named_parameters())

        def backbone_weights():
            """Consume the checkpoint once while loading the trained head parameters."""
            for name, value in weights:
                if name.startswith("decision_head."):
                    suffix = name.removeprefix("decision_head.")
                    default_weight_loader(parameters[suffix], value)
                    loaded_head.add(f"pooler.decision_head.{suffix}")
                else:
                    yield name.replace("backbone.", "model.", 1), value

        loaded = super().load_weights(backbone_weights())
        return loaded | loaded_head
