# Installation

## Transformers

```bash
uv pip install social-jev
```

The default backend is Transformers. Loading a checkpoint uses CPU placement unless
you supply native Transformers placement options or move the loaded model.
The first load downloads the model and tokenizer from Hugging Face.

## Optional vLLM backend

```bash
uv pip install 'social-jev[vllm]'
```

This extra installs the vLLM version supported by SocialJev and HTTPX for serving.
Use a supported Linux GPU environment. Our native vLLM execution has been exercised
on NVIDIA GH200 hardware; the custom decision-head adapter is not supported by
vLLM-Metal's Apple Silicon pooling backend.

Installing the regular package does not install vLLM or documentation tooling.
The Python requirement is 3.12 or newer; resolve dependencies for the hardware and
Python environment you intend to use.

## Use a specific release

```bash
uv pip install 'social-jev==0.1.0a2'
```

The selected documentation version identifies the package release it describes.
The preview model is a separate, evolving HF repository. Native `revision` can
select a model repository tag; use the same checkpoint revision when comparing
execution backends.

Continue with the [quickstart](index.md) or [serving guide](serving.md).
