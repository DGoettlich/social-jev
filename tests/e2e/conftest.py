import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from social_jev import Processor
from social_jev.config import SocialJevConfig
from social_jev.transformers import LinearModel


@pytest.fixture
def checkpoint(tmp_path):
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
        vocab_size=len(words),
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=256,
        pad_token_id=0,
    )
    model = LinearModel(config)
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
            parameter.copy_(torch.sin(values + index) * 0.05)
    model.save_pretrained(tmp_path)
    return tmp_path
