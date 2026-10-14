import asyncio
import importlib.util
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from social_jev import Processor, Query, Request, SocialJev, server


def test_async_requests_keep_loop_responsive_and_stop_owned_server(checkpoint, monkeypatch):  # noqa: PLR0915 -- Exercise one complete serving lifecycle.
    real_popen = subprocess.Popen
    real_find_spec = importlib.util.find_spec
    commands = []

    def launch(command, **kwargs):
        commands.append(command)
        port = command[command.index("--port") + 1]
        fixture = Path(__file__).parent / "fixtures/pooling_service.py"
        return real_popen([sys.executable, str(fixture), str(checkpoint), port], **kwargs)

    monkeypatch.setattr(server.subprocess, "Popen", launch)
    monkeypatch.setattr(
        server.importlib.util,
        "find_spec",
        lambda name: object() if name == "vllm" else real_find_spec(name),
    )
    offline = SocialJev.from_pretrained(checkpoint)
    request = Request(state="S", queries=[Query(text="Q", options=["A", "B"])])
    expected = offline.query(request)
    model = SocialJev.from_pretrained(checkpoint, backend="vllm", serve=True)
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
        with pytest.raises(RuntimeError, match=r"await model\.aquery"):
            model.query(request)
        asyncio.run(concurrent())
        assert model.engine.process.poll() is not None
        assert not model.engine.thread.is_alive()
        assert model.engine.client.is_closed
        assert commands[0][commands[0].index("--host") + 1] == "127.0.0.1"
        model.stop()
    finally:
        release.set()
        model.stop()
        offline.stop()
