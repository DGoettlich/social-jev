"""Exercise the release documentation and wheel builds through their CLI interfaces."""

import importlib.util
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mkdocs") is None,
    reason="Run with uv run --group docs to build documentation",
)


def test_release_docs_and_wheel_keep_tooling_separate(tmp_path):
    """Build versioned public docs and verify the public wheel excludes docs dependencies."""
    root = Path(__file__).resolve().parents[2]
    site = tmp_path / "site"
    subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "--site-dir", str(site)],
        cwd=root,
        env={**os.environ, "DOCS_VERSION": "v9.8.7"},
        check=True,
    )
    homepage = (site / "index.html").read_text()
    assert 'data-docs-version="v9.8.7"' in homepage
    assert 'href="https://social-jev.github.io/docs/v9.8.7/"' in homepage
    for page in ("concepts/index.html", "inference/index.html", "api/index.html"):
        assert (site / page).is_file()
    assert not (site / "reference").exists()
    assert not (site / "roadmap").exists()
    reference = (site / "api/index.html").read_text()
    assert "Processor" not in reference
    assert "VLLM_BATCH_INVARIANT" in reference
    assert "Additional backend arguments" in reference
    for name in ("SocialJev", "Request", "Query", "Result"):
        assert name in reference
    assert (site / "search/search_index.json").is_file()
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(tmp_path)], cwd=root, check=True)
    wheel = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read(
            next(name for name in archive.namelist() if name.endswith("/METADATA"))
        ).decode()
    dependencies = [
        line.lower() for line in metadata.splitlines() if line.startswith("Requires-Dist:")
    ]
    assert not any("mkdocs" in line or "mkdocstrings" in line for line in dependencies)
    assert "Provides-Extra: docs" not in metadata
