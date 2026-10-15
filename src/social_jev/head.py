"""Scalar decision heads matching the linear and MLP architectures in mert."""

from dataclasses import dataclass
from itertools import pairwise

from torch import nn


@dataclass
class DecisionHeadConfig:
    """Saved architecture settings for a scalar linear or MLP head."""

    type: str = "linear"
    """Trained head architecture: linear or mlp."""
    dims: list[int] | None = None
    """Output widths of successive linear layers, ending in one scalar."""
    activation: str = "gelu"
    """Activation between hidden layers."""
    layernorm: bool = False
    """Whether later hidden layers use the normalization placement from mert."""

    def __post_init__(self):
        """Resolve defaults and require a supported scalar architecture."""
        defaults = {"linear": [1], "mlp": [512, 1]}
        if self.type not in defaults:
            message = "SocialJev supports linear and mlp decision heads"
            raise ValueError(message)
        if self.dims is None:
            self.dims = defaults[self.type].copy()
        assert self.dims
        assert all(width > 0 for width in self.dims)
        assert self.activation in {"gelu", "relu", "silu"}
        assert (
            self.dims == [1] if self.type == "linear" else len(self.dims) > 1 and self.dims[-1] == 1
        )

    def layers(self, input_dim: int) -> nn.Sequential:
        """Reconstruct the trained layers using mert's widths and normalization placement."""
        activation = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[self.activation]
        assert self.dims is not None
        widths = [input_dim, *self.dims]
        layers = []
        for index, (before, after) in enumerate(pairwise(widths)):
            layers.append(nn.Linear(before, after))
            if index < len(widths) - 2:
                if self.layernorm and index > 0:
                    layers.append(nn.LayerNorm(after))
                layers.append(activation())
        return nn.Sequential(*layers)


class ScalarHead(nn.Module):
    """Apply a trained linear or MLP network to candidate hidden states.

    Args:
        config: Saved architecture settings.
        input_dim: Backbone hidden-state width.

    Examples:
        >>> head = ScalarHead(DecisionHeadConfig(), 16)
        >>> head.network[0].out_features
        1
    """

    def __init__(self, config: DecisionHeadConfig, input_dim: int):
        """Construct the network with checkpoint-compatible parameter names."""
        super().__init__()
        self.network = config.layers(input_dim)

    def forward(self, options):
        """Return one scalar logit per candidate hidden state."""
        return self.network(options).squeeze(-1)
