import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config

from social_jev import Processor, SocialJev
from social_jev.config import SocialJevConfig


@pytest.fixture
def model(tmp_path):
    words = [
        "[PAD]",
        "[UNK]",
        "<|im_start|>",
        "<|im_end|>",
        "state",
        "query",
        "option",
        "example",
        "answer",
        "S",
        "Q",
        "R",
        "A",
        "B",
        "C",
        "A:",
        "B:",
        "C:",
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
        backbone_config=Qwen3Config(
            vocab_size=len(words),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=256,
        ),
        option_codes="ABC",
    )
    instance = SocialJev(config)
    with torch.no_grad():
        for index, parameter in enumerate(instance.parameters()):
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
            parameter.copy_(torch.sin(values + index) * 0.05)
    instance.save_pretrained(tmp_path)
    return SocialJev.from_pretrained(tmp_path, local_files_only=True).eval()
