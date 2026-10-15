import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from social_jev import Processor
from social_jev.config import SocialJevConfig
from social_jev.transformers import SocialJevModel


@pytest.fixture(
    params=[
        {"type": "linear"},
        {"type": "mlp", "dims": [32, 1], "activation": "gelu"},
        {"type": "mlp", "dims": [32, 16, 1], "activation": "silu", "layernorm": True},
        {"type": "mlp", "dims": [32, 1], "activation": "relu"},
    ],
    ids=["linear", "mlp-gelu", "mlp-silu-layernorm", "mlp-relu"],
)
def checkpoint(tmp_path, request):
    words = [
        "[PAD]",
        "[UNK]",
        "<|im_start|>",
        "<|im_end|>",
        "state",
        "query",
        "option",
        "S",
        "Q",
        "R",
        "A",
        "B",
        "C",
    ]
    backend = Tokenizer(
        models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    tokenizer.chat_template = Processor.template()
    tokenizer.save_pretrained(tmp_path)
    config = SocialJevConfig(
        decision_head=request.param,
        backbone_config={
            "model_type": "qwen3",
            "vocab_size": len(words),
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "max_position_embeddings": 256,
            "pad_token_id": 0,
        },
    )
    model = SocialJevModel(config)
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
            parameter.copy_(torch.sin(values + index) * 0.05)
    model.save_pretrained(tmp_path)
    return tmp_path
