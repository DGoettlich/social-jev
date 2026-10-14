"""Owned localhost vLLM serving with reusable asynchronous HTTP connections."""

import asyncio
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http import HTTPStatus

import httpx


class Server:
    """Own a localhost vLLM server and an internal HTTP event loop.

    Args:
        repo: Self-contained linear checkpoint to serve.

    Examples:
        >>> server = Server(repo)  # doctest: +SKIP
        >>> server.stop()  # doctest: +SKIP
    """

    def __init__(self, repo):
        """Start the server, wait for readiness, and create one reusable client."""
        if importlib.util.find_spec("vllm") is None:
            message = 'Install the optional backend with uv pip install "social-jev[vllm]" on Linux'
            raise ImportError(message)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            str(repo),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--served-model-name",
            "social-jev",
            "--runner",
            "pooling",
            "--hf-overrides",
            json.dumps({"architectures": ["SocialJevPoolingModel"], "is_causal": True}),
            "--pooler-config",
            json.dumps({"seq_pooling_type": "LAST", "use_activation": False}),
            "--enable-prefix-caching",
            "--no-enable-chunked-prefill",
            "--gpu-memory-utilization",
            "0.8",
        ]
        self.process = subprocess.Popen(
            command,
            start_new_session=True,
            env={**os.environ, "VLLM_PLUGINS": "social_jev"},
        )
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.client = None
        try:
            self.wait_ready()
            asyncio.run_coroutine_threadsafe(self.connect(), self.loop).result()
        except BaseException:
            self.stop()
            raise

    def wait_ready(self):
        """Wait for model readiness while detecting startup failures."""
        deadline = time.monotonic() + 300
        with httpx.Client(base_url=self.url, timeout=1) as probe:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    message = "vLLM exited during startup; see its server logs"
                    raise RuntimeError(message)
                try:
                    if probe.get("/health").status_code == HTTPStatus.OK:
                        return
                except httpx.TransportError:
                    pass  # Connection refusal is expected while the model starts.
                time.sleep(0.1)
        message = "vLLM did not become ready within 300 seconds"
        raise TimeoutError(message)

    async def connect(self):
        """Create the HTTP client on the loop that owns its connections."""
        self.client = httpx.AsyncClient(base_url=self.url, timeout=None)

    async def request(self, inputs: list[list[int]]) -> list[float]:
        """Submit token sequences and restore the response's candidate order."""
        assert self.client is not None
        response = await self.client.post(
            "/pooling",
            json={
                "model": "social-jev",
                "task": "embed",
                "input": inputs,
                "add_special_tokens": False,
                "use_activation": False,
            },
        )
        response.raise_for_status()
        entries = sorted(response.json()["data"], key=lambda entry: entry["index"])
        return [entry["data"][0] for entry in entries]

    async def ascore(self, inputs: list[list[int]]) -> list[float]:
        """Await HTTP execution without blocking the caller's event loop."""
        if not inputs:
            return []
        future = asyncio.run_coroutine_threadsafe(self.request(inputs), self.loop)
        return await asyncio.wrap_future(future)

    def stop(self):
        """Close HTTP connections and stop only the owned server and workers."""
        if self.loop.is_running():
            if self.client is not None:
                asyncio.run_coroutine_threadsafe(self.client.aclose(), self.loop).result()
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join()
            self.loop.close()
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
