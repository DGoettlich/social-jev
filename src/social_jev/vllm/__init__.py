"""Optional vLLM execution settings shared by offline and online inference."""


def engine_settings(**kwargs) -> dict:
    """Forward execution options while fixing the method's causal pooling contract."""
    settings = {"enable_prefix_caching": True, "gpu_memory_utilization": 0.8, **kwargs}
    fixed = {
        "runner": "pooling",
        "hf_overrides": {"architectures": ["SocialJevPoolingModel"], "is_causal": True},
        "pooler_config": {"seq_pooling_type": "LAST", "use_activation": False},
        "enable_chunked_prefill": False,
    }
    if fixed.keys() & kwargs.keys():
        message = "SocialJev owns the model architecture and causal pooling settings"
        raise ValueError(message)
    settings.update(fixed)
    return settings
