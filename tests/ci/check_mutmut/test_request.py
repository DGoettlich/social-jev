"""Exercise mutation requests through their event input and external GitHub CLI."""

import json
import os
import sys
from pathlib import Path

import pytest

from ci.check_mutmut.request import main


@pytest.fixture
def github_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/project")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "issue_comment")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(tmp_path / "event.json"))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    (tmp_path / "event.json").write_text(
        json.dumps(
            {
                "action": "created",
                "comment": {"id": 7, "body": "/mutmut", "user": {"login": "writer"}},
                "issue": {"number": 14, "pull_request": {}},
            }
        )
    )
    (tmp_path / "responses.json").write_text(
        json.dumps(
            {
                "collaborators/writer/permission": {"permission": "write"},
                "pulls/14": {
                    "state": "open",
                    "head": {
                        "sha": "requested-version",
                        "ref": "feature/change",
                        "repo": {"full_name": "owner/project"},
                    },
                },
            }
        )
    )
    tool = tmp_path / "gh"
    tool.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "endpoint = sys.argv[2].removeprefix('repos/owner/project/')\n"
        "if '--input' in sys.argv:\n"
        "    with pathlib.Path('writes.jsonl').open('a') as output:\n"
        "        output.write(json.dumps([endpoint, json.load(sys.stdin)]) + '\\n')\n"
        "    if os.environ.get('FAIL_REACTION') and endpoint.endswith('/reactions'): sys.exit(1)\n"
        "else:\n"
        "    print(json.dumps(json.loads(pathlib.Path('responses.json').read_text())[endpoint]))\n"
    )
    tool.chmod(0o755)
    return tmp_path


def test_command_dispatches_the_current_pr_version(github_api: Path) -> None:
    assert main() == 0
    writes = [json.loads(line) for line in (github_api / "writes.jsonl").read_text().splitlines()]
    assert [
        "actions/workflows/ci.yaml/dispatches",
        {
            "ref": "feature/change",
            "inputs": {"mutation_pr": "14", "mutation_revision": "requested-version"},
        },
    ] in writes


@pytest.mark.parametrize(
    "changes",
    [
        {"action": "edited"},
        {"comment": {"body": "please /mutmut"}},
        {"comment": {"body": "/mutmut fresh"}},
        {"issue": {"number": 14}},
    ],
)
def test_only_new_exact_pr_commands_can_dispatch(github_api: Path, changes: dict) -> None:
    event = github_api / "event.json"
    event.write_text(json.dumps(json.loads(event.read_text()) | changes))
    assert main() == 0
    assert not (github_api / "writes.jsonl").exists()


@pytest.mark.parametrize("permission", ["read", "triage", "none"])
def test_repository_write_access_is_required(github_api: Path, permission: str) -> None:
    responses = github_api / "responses.json"
    data = json.loads(responses.read_text())
    data["collaborators/writer/permission"]["permission"] = permission
    responses.write_text(json.dumps(data))
    assert main() == 1
    assert not (github_api / "writes.jsonl").exists()
    assert "write access" in (github_api / "summary.md").read_text()


@pytest.mark.parametrize(
    "changes",
    [
        {"state": "closed"},
        {"head": {"repo": {"full_name": "fork/project"}}},
        {"head": {"repo": None}},
    ],
)
def test_only_open_same_repository_prs_can_run(github_api: Path, changes: dict) -> None:
    responses = github_api / "responses.json"
    data = json.loads(responses.read_text())
    data["pulls/14"].update(changes)
    responses.write_text(json.dumps(data))
    assert main() == 1
    assert not (github_api / "writes.jsonl").exists()


@pytest.mark.parametrize(
    ("requested", "checked_out", "expected"),
    [
        ("requested-version", "requested-version", 0),
        ("previous-version", "requested-version", 1),
        ("requested-version", "previous-version", 1),
        ("", "requested-version", 1),
    ],
)
def test_worker_requires_the_requested_and_current_pr_version(
    github_api: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    checked_out: str,
    expected: int,
) -> None:
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_SHA", checked_out)
    (github_api / "event.json").write_text(
        json.dumps(
            {
                "inputs": {"mutation_pr": "14", "mutation_revision": requested},
            }
        )
    )
    assert main() == expected
    assert not (github_api / "writes.jsonl").exists()


def test_github_failure_cannot_dispatch_or_expose_api_output(github_api: Path) -> None:
    (github_api / "gh").write_text("#!/bin/sh\necho sensitive-response >&2\nexit 1\n")
    assert main() == 1
    assert not (github_api / "writes.jsonl").exists()
    summary = (github_api / "summary.md").read_text()
    assert "GitHub request failed" in summary
    assert "sensitive-response" not in summary


def test_acknowledgement_failure_does_not_claim_the_run_was_not_started(
    github_api: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAIL_REACTION", "1")
    assert main() == 1
    writes = (github_api / "writes.jsonl").read_text()
    assert "actions/workflows/ci.yaml/dispatches" in writes
    summary = (github_api / "summary.md").read_text()
    assert "not started" not in summary
    assert "before retrying" in summary
