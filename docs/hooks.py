"""Set canonical documentation URLs from the release being built."""

import os
import tomllib
from pathlib import Path


def on_config(config):
    """Use DOCS_VERSION for release builds and the package version locally."""
    project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())[
        "project"
    ]
    version = os.environ.get("DOCS_VERSION") or f"v{project['version']}"
    config.site_url = f"https://social-jev.github.io/docs/{version}/"
    config.extra["package_version"] = version
    return config
