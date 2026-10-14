"""Native vLLM Qwen3 execution with a replicated trained linear pooler."""

from torch import nn
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


class LinearPooler(Pooler):
    """Apply the trained linear head to completed candidates' terminal tokens.

    Args:
        hidden_size: Qwen3 hidden-state width.

    Examples:
        >>> pooler = LinearPooler(16)  # doctest: +SKIP
    """

    def __init__(self, hidden_size: int):
        """Create the replicated scalar head on the worker's allocation device."""
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def get_supported_tasks(self):
        """Expose scalar logits through vLLM's pooling transport."""
        return {"embed"}

    def forward(self, hidden_states, pooling_metadata):
        """Select the final token and return raw logits without normalization."""
        cursor = pooling_metadata.get_pooling_cursor()
        hidden = hidden_states[cursor.last_token_indices_gpu]
        logits = self.linear(hidden.to(self.linear.weight.dtype)).float()
        return list(logits.unbind())


class PoolingModel(as_embedding_model(Qwen3ForCausalLM)):
    """Use vLLM's native Qwen3 backbone and the checkpoint's linear head.

    Args:
        vllm_config: Engine-supplied model configuration.
        prefix: Engine-supplied parameter prefix.

    Examples:
        >>> from social_jev.plugin import register
        >>> register()  # doctest: +SKIP
    """

    def __init__(self, *, vllm_config, prefix=""):
        """Replace the embedding wrapper's default pooler with scalar scoring."""
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.pooler = LinearPooler(vllm_config.model_config.hf_config.hidden_size)

    def load_weights(self, weights):
        """Route backbone weights to native loading and head weights to the pooler."""
        loaded_head = set()

        def backbone_weights():
            """Consume the checkpoint once while loading the two head parameters."""
            for name, value in weights:
                if name.startswith("decision_head."):
                    suffix = name.removeprefix("decision_head.")
                    default_weight_loader(getattr(self.pooler.linear, suffix), value)
                    loaded_head.add(f"pooler.linear.{suffix}")
                else:
                    yield name.replace("backbone.", "model.", 1), value

        loaded = super().load_weights(backbone_weights())
        return loaded | loaded_head
