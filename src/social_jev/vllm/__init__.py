"""Optional vLLM execution settings shared by offline and online inference."""

from dataclasses import asdict, is_dataclass


def engine_settings(**kwargs) -> dict:
    """Combine caller settings with the required classification integration."""
    settings = dict(kwargs)
    settings["hf_overrides"] = {
        **(settings.get("hf_overrides") or {}),
        "architectures": ["SocialJevPoolingModel"],
        "is_causal": True,
    }
    pooler = settings.get("pooler_config") or {}
    if is_dataclass(pooler):
        pooler = asdict(pooler)
    settings["pooler_config"] = {
        **pooler,
        "seq_pooling_type": "LAST",
        "use_activation": False,
    }
    settings["runner"] = "pooling"
    return settings
