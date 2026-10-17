"""Owned localhost vLLM serving with reusable asynchronous HTTP connections."""

import asyncio
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http import HTTPStatus

import httpx

from . import engine_settings


def cli_arguments(settings: dict) -> list[str]:
    """Translate native execution settings to vLLM CLI arguments."""
    arguments = []
    for name, value in settings.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            arguments.append(flag if value else "--no-" + name.replace("_", "-"))
        elif isinstance(value, dict):
            arguments.extend([flag, json.dumps(value)])
        elif isinstance(value, (list, tuple)):
            arguments.extend([flag, *map(str, value)])
        elif value is not None:
            arguments.extend([flag, str(value)])
    return arguments


class Server:
    """Launch a managed vLLM HTTP server for synchronous or asynchronous inference.

    Args:
        repo: Self-contained scalar checkpoint to serve.

    Examples:
        >>> server = Server(repo)  # doctest: +SKIP
        >>> server.stop()  # doctest: +SKIP
    """

    def __init__(self, repo, **kwargs):
        if importlib.util.find_spec("vllm") is None:
            message = 'Install the optional backend with uv pip install "social-jev[vllm]" on Linux'
            raise ImportError(message)
        from vllm.entrypoints.launchers.cli_args import (  # noqa: PLC0415  # ty: ignore[unresolved-import]
            make_arg_parser,
        )
        from vllm.utils.argparse_utils import (  # noqa: PLC0415  # ty: ignore[unresolved-import]
            FlexibleArgumentParser,
        )

        settings = engine_settings(**kwargs)
        settings["served_model_name"] = "social-jev"
        arguments = [str(repo), *cli_arguments(settings)]
        parsed = make_arg_parser(FlexibleArgumentParser()).parse_args(arguments)
        host = parsed.host
        if host in {None, "", "0.0.0.0"}:
            host = "127.0.0.1"
        elif host == "::":
            host = "::1"
        address = f"[{host}]" if ":" in host else host
        self.url = f"http://{address}:{parsed.port}"
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            *arguments,
        ]
        self.server_process = subprocess.Popen(command, start_new_session=True)
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
        with httpx.Client(base_url=self.url, timeout=1) as probe:
            while True:
                if self.server_process.poll() is not None:
                    message = "vLLM exited during startup; see its server logs"
                    raise RuntimeError(message)
                try:
                    if probe.get("/health").status_code == HTTPStatus.OK:
                        return
                except httpx.TransportError:
                    pass  # Connection refusal is expected while the model starts.
                time.sleep(0.1)

    async def connect(self):
        """Create the HTTP client on the loop that owns its connections."""
        self.client = httpx.AsyncClient(base_url=self.url, timeout=None)

    async def submit_as_query_option_sequences(self, inputs: list[list[int]]) -> list[float]:
        assert self.client is not None
        response = await self.client.post(
            "/pooling",
            json={
                "model": "social-jev",
                "task": "embed",
                "input": inputs,
            },
        )
        response.raise_for_status()
        entries = sorted(response.json()["data"], key=lambda entry: entry["index"])
        return [entry["data"][0] for entry in entries]

    def process(self, inputs: list[list[int]]) -> list[float]:
        """Process token sequences synchronously and return scalar scores in input order."""
        if not inputs:
            return []
        return asyncio.run_coroutine_threadsafe(
            self.submit_as_query_option_sequences(inputs), self.loop
        ).result()

    async def aprocess(self, inputs: list[list[int]]) -> list[float]:
        """Process token sequences asynchronously and return scalar scores in input order."""
        if not inputs:
            return []
        future = asyncio.run_coroutine_threadsafe(
            self.submit_as_query_option_sequences(inputs), self.loop
        )
        return await asyncio.wrap_future(future)

    def stop(self):
        """Close HTTP connections and stop only the owned server and workers."""
        if self.loop.is_running():
            if self.client is not None:
                asyncio.run_coroutine_threadsafe(self.client.aclose(), self.loop).result()
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join()
            self.loop.close()
        if self.server_process.poll() is None:
            os.killpg(self.server_process.pid, signal.SIGTERM)
            self.server_process.wait()
