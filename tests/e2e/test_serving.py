import argparse
import asyncio
import importlib.util
import json
import socket
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from social_jev import Processor, Query, Request, SocialJev
from social_jev.vllm import server


@pytest.mark.parametrize("explicit_address", [False, True])
def test_async_requests_keep_loop_responsive_and_stop_owned_server(  # noqa: C901, PLR0915 -- Exercise one complete serving lifecycle.
    checkpoint, monkeypatch, explicit_address
):
    real_popen = subprocess.Popen
    real_find_spec = importlib.util.find_spec
    commands = []
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    # Substitute only the optional vLLM parser; exercise the real client and child process.
    def make_arg_parser(parser):
        parser.add_argument("repo")
        parser.add_argument("--host", default=None)
        parser.add_argument("--port", type=int, default=port)
        return SimpleNamespace(parse_args=lambda args: parser.parse_known_args(args)[0])

    monkeypatch.setitem(
        sys.modules,
        "vllm.entrypoints.launchers.cli_args",
        SimpleNamespace(make_arg_parser=make_arg_parser),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.utils.argparse_utils",
        SimpleNamespace(FlexibleArgumentParser=argparse.ArgumentParser),
    )

    def launch(command, **kwargs):
        commands.append(command)
        fixture = Path(__file__).parent / "fixtures/pooling_service.py"
        return real_popen([sys.executable, str(fixture), str(checkpoint), str(port)], **kwargs)

    monkeypatch.setattr(server.subprocess, "Popen", launch)
    monkeypatch.setattr(
        server.importlib.util,
        "find_spec",
        lambda name: object() if name == "vllm" else real_find_spec(name),
    )
    offline = SocialJev.from_pretrained(checkpoint)
    request = Request(state="S", queries=[Query(text="Q", options=["A", "B"])])
    expected = offline.query(request)
    address = (
        {
            "host": "0.0.0.0",
            "port": port,
            "gpu_memory_utilization": 0.9,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
        }
        if explicit_address
        else {}
    )
    model = SocialJev.from_pretrained(
        checkpoint,
        backend="vllm",
        serve=True,
        tensor_parallel_size=4,
        pipeline_parallel_size=2,
        hf_overrides={"attention_dropout": 0.0},
        pooler_config={"normalize": False},
        **address,
    )
    entered = threading.Event()
    release = threading.Event()
    original_encode = Processor.encode

    def slow_encode(self, requests):
        entered.set()
        if not release.wait(timeout=10):
            message = "Event loop did not release CPU preparation"
            raise TimeoutError(message)
        return original_encode(self, requests)

    monkeypatch.setattr(Processor, "encode", slow_encode)

    async def concurrent():
        semaphore = asyncio.Semaphore(2)

        async def submit():
            async with semaphore:
                return await model.aquery(request)

        tasks = [asyncio.create_task(submit()) for _ in range(4)]
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            # This coroutine must run while encode() is waiting in a worker thread.
            release.set()
            results = await asyncio.gather(*tasks)
            assert all(result == expected for result in results)
            assert await model.aquery([]) == []
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            # Synchronous stop is also valid inside an active caller event loop.
            model.stop()

    try:
        release.set()
        assert model.query(request) == expected
        assert model.query([]) == []
        release.clear()
        asyncio.run(concurrent())
        assert model.engine.process.poll() is not None
        assert not model.engine.thread.is_alive()
        assert model.engine.client.is_closed
        command = commands[0]
        assert command[command.index("--tensor-parallel-size") + 1] == "4"
        assert command[command.index("--pipeline-parallel-size") + 1] == "2"
        assert model.engine.url == f"http://127.0.0.1:{port}"
        assert ("--host" in command) is explicit_address
        assert ("--port" in command) is explicit_address
        assert command[command.index("--served-model-name") + 1] == "social-jev"
        assert ("--gpu-memory-utilization" in command) is explicit_address
        assert ("--no-enable-prefix-caching" in command) is explicit_address
        assert ("--no-enable-chunked-prefill" in command) is explicit_address
        assert "--enable-prefix-caching" not in command
        assert "--enable-chunked-prefill" not in command
        overrides = json.loads(command[command.index("--hf-overrides") + 1])
        assert overrides == {
            "attention_dropout": 0.0,
            "architectures": ["SocialJevPoolingModel"],
            "is_causal": True,
        }
        pooler = json.loads(command[command.index("--pooler-config") + 1])
        assert pooler == {"normalize": False, "seq_pooling_type": "LAST", "use_activation": False}
        model.stop()
    finally:
        release.set()
        model.stop()
        offline.stop()
