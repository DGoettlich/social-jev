"""Register the native pooling implementation in vLLM worker processes."""


def register():
    """Make the fixed SocialJev architecture available to vLLM."""
    from vllm import (  # ty: ignore[unresolved-import]  # noqa: PLC0415
        ModelRegistry,
    )

    ModelRegistry.register_model("SocialJevPoolingModel", "social_jev.pooling:PoolingModel")
