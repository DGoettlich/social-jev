# Checkpoint format

A complete HF export contains:

```text
config.json
model.safetensors                  # or indexed safetensors shards
tokenizer.json
tokenizer_config.json
chat_template.jinja
```

The tokenizer assets depend on the tokenizer implementation; the preview uses the
files shown above. The package loads its own model implementation, so loading
through SocialJev does not require trusting remote Python code.

## Architecture and weights

The root config uses `model_type="social_jev"`, with the native Qwen3 configuration
under `backbone_config` and the head architecture under `decision_head`.

A linear head:

```json
{"type": "linear", "dims": [1], "activation": "gelu", "layernorm": false}
```

An MLP head:

```json
{"type": "mlp", "dims": [512, 1], "activation": "gelu", "layernorm": false}
```

Widths specify successive output dimensions, ending in one scalar. Supported
activations are GELU, ReLU and SiLU. These settings describe how the head was
trained; changing them after export does not convert its weights.

Weight names retain `backbone.*` and `decision_head.network.*`. For a linear head,
the readout uses `decision_head.network.0.weight` and its bias. vLLM routes the
backbone to native Qwen3 loading and the decision-head weights to its custom pooler.

## Tokenizer and template

The tokenizer must save its padding token. Transformers reads the ID from the
loaded tokenizer; the backbone config need not duplicate it.

The exported template must work with `tokenizer.apply_chat_template(messages)`
without external variables. It uses state, query and option roles, and closes each
message with `<|im_end|>`. Embed saved role delimiter values into the template
rather than relying on model-config variables reaching the tokenizer.

`social-jev/social-jev-0.6B-preview` has been smoke-tested unchanged through all
three execution paths. A model revision is independent of the package release;
pin a repository tag when you need reproducible checkpoint selection.
