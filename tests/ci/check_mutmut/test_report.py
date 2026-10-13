"""Exercise the report command with an external mutation-tool stub."""

import json
import os
import sys
from pathlib import Path

import pytest

from ci.check_mutmut.report import main


@pytest.fixture
def mutation_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("MUTATION_OUTCOME", "success")
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/project")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    tool = tmp_path / "mutmut"
    tool.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, shutil, sys\n"
        "if sys.argv[1] == 'export-cicd-stats':\n"
        "    pathlib.Path('mutants').mkdir(exist_ok=True)\n"
        "    shutil.copyfile('stats.json', 'mutants/mutmut-cicd-stats.json')\n"
        "elif sys.argv[1] == 'show':\n"
        "    print(json.loads(pathlib.Path('diffs.json').read_text())[sys.argv[2]])\n"
        "else:\n"
        "    print(pathlib.Path('results.txt').read_text())\n"
    )
    tool.chmod(0o755)
    (tmp_path / "results.txt").write_text("package.x_check__mutmut_1: survived\n")
    (tmp_path / "diffs.json").write_text(
        json.dumps(
            {
                "package.x_check__mutmut_1": (
                    "--- check.py\n+++ check.py\n-    return value >= 3\n+    return value > 3\n"
                ),
            }
        )
    )
    stats = tmp_path / "stats.json"
    stats.write_text(
        json.dumps(
            {
                "total": 5,
                "killed": 3,
                "survived": 1,
                "no_tests": 1,
                "skipped": 0,
                "suspicious": 0,
                "timeout": 0,
                "check_was_interrupted_by_user": 0,
                "segfault": 0,
            }
        )
    )
    return stats


def test_survivors_are_advisory_and_visible(mutation_tool: Path) -> None:
    assert main() == 0
    report = json.loads((mutation_tool.parent / "mutation-report/report.json").read_text())
    assert report["status"] == "complete"
    assert report["counts"]["survived"] == 1
    assert report["counts"]["no_tests"] == 1
    survivor = report["survivors"][0]
    assert survivor["name"] == "package.x_check__mutmut_1"
    assert "-    return value >= 3" in survivor["diff"]
    summary = Path("mutation-report/summary.md").read_text()
    assert "package.check · mutation 1" in summary
    assert "+    return value &gt; 3" in summary


@pytest.mark.parametrize(
    ("changes", "outcome", "status"),
    [
        ({"total": 6}, "success", "incomplete"),
        ({"killed": 2, "timeout": 1}, "success", "failed"),
        ({"killed": 2, "suspicious": 1}, "success", "failed"),
        ({"killed": 2, "segfault": 1}, "success", "failed"),
        ({"killed": 2, "check_was_interrupted_by_user": 1}, "success", "incomplete"),
        ({}, "failure", "failed"),
        ({}, "cancelled", "incomplete"),
    ],
)
def test_partial_and_failed_runs_cannot_appear_complete(
    mutation_tool: Path,
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, int],
    outcome: str,
    status: str,
) -> None:
    stats = json.loads(mutation_tool.read_text()) | changes
    mutation_tool.write_text(json.dumps(stats))
    monkeypatch.setenv("MUTATION_OUTCOME", outcome)
    assert main() == 1
    report = json.loads(Path("mutation-report/report.json").read_text())
    assert report["status"] == status
    assert report["counts"]["killed"] == stats["killed"]
    assert report["counts"]["timeout"] == stats["timeout"]


@pytest.mark.parametrize(
    "data",
    [
        None,
        "not json",
        "[]",
        "{}",
        '{"total": true}',
        '{"total": -1}',
    ],
)
def test_missing_and_malformed_data_produces_failure_without_invented_counts(
    mutation_tool: Path,
    data: str | None,
) -> None:
    if data is None:
        mutation_tool.unlink()
    else:
        mutation_tool.write_text(data)
    assert main() == 1
    report = json.loads(Path("mutation-report/report.json").read_text())
    assert report["status"] == "failed"
    assert report["counts"] is None
    assert "Counts unavailable" in Path("mutation-report/summary.md").read_text()


def test_complete_cached_result_is_reused_without_running_tool(
    mutation_tool: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert main() == 0
    (mutation_tool.parent / "mutmut").unlink()
    monkeypatch.setenv("MUTATION_REUSE", "true")
    monkeypatch.setenv("GITHUB_RUN_ID", "456")
    assert main() == 0
    report = json.loads(Path("mutation-report/report.json").read_text())
    assert report["reused"] is True
    assert report["producer"].endswith("/123")
    assert "+    return value > 3" in report["survivors"][0]["diff"]
    assert "reused result" in Path("mutation-report/summary.md").read_text()


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "incomplete"},
        {"version": 999},
        {"version": 1},
        {"survivors": [{"name": "package.check"}]},
        {"survivors": [{"name": "package.check", "diff": " "}]},
        {"survivors": [{"name": "package.check", "diff": 42}]},
        {"survivors": [{"name": "", "diff": "+ changed"}]},
        {"survivors": [{"name": "package.check", "diff": "x" * 4001}]},
        {"counts": {}},
        {"survivors": [42]},
        {"producer": "https://untrusted.invalid/"},
    ],
)
def test_invalid_cached_report_is_rejected(
    mutation_tool: Path,
    monkeypatch: pytest.MonkeyPatch,
    changes: dict,
) -> None:
    assert main() == 0
    report_path = mutation_tool.parent / "mutation-report/report.json"
    report_path.write_text(json.dumps(json.loads(report_path.read_text()) | changes))
    monkeypatch.setenv("MUTATION_REUSE", "true")
    assert main() == 1
    assert json.loads(report_path.read_text())["status"] == "failed"


def test_survivor_text_is_bounded_and_escaped(mutation_tool: Path) -> None:
    stats = json.loads(mutation_tool.read_text()) | {"total": 34, "survived": 30}
    mutation_tool.write_text(json.dumps(stats))
    Path("results.txt").write_text("<script>@someone: survived\n" * 30)
    Path("diffs.json").write_text(json.dumps({"<script>@someone": "- old\n+ <script>@reviewer"}))
    assert main() == 0
    summary = Path("mutation-report/summary.md").read_text()
    assert "<script>" not in summary
    assert "@someone" not in summary
    assert summary.count("&#64;someone") == 20
    assert "@reviewer" not in summary
    assert summary.count("&lt;script&gt;&#64;reviewer") == 20


@pytest.mark.parametrize("changes", [{"total": 4}, {"killed": -1}, {"killed": True}])
def test_inconsistent_counts_cannot_be_reported_as_success(
    mutation_tool: Path,
    changes: dict,
) -> None:
    mutation_tool.write_text(json.dumps(json.loads(mutation_tool.read_text()) | changes))
    assert main() == 1
    report = json.loads(Path("mutation-report/report.json").read_text())
    assert report["status"] == "failed"
    assert report["counts"] is None


def test_missing_survivor_details_cannot_be_cached_as_complete(mutation_tool: Path) -> None:
    (mutation_tool.parent / "results.txt").write_text("")
    assert main() == 1
    assert json.loads(Path("mutation-report/report.json").read_text())["status"] == "failed"


@pytest.mark.parametrize(
    "changes",
    [
        {"counts": {"killed": 2, "not_checked": 1}},
        {"counts": {"total": 6}},
        {"survivors": [{"name": "x" * 501, "diff": "+ changed"}]},
        {"survivors": []},
        {"version": True},
    ],
)
def test_cache_cannot_hide_partial_or_inconsistent_results(
    mutation_tool: Path,
    monkeypatch: pytest.MonkeyPatch,
    changes: dict,
) -> None:
    assert main() == 0
    report_path = mutation_tool.parent / "mutation-report/report.json"
    report = json.loads(report_path.read_text())
    if "counts" in changes:
        report["counts"].update(changes["counts"])
    else:
        report.update(changes)
    report_path.write_text(json.dumps(report))
    monkeypatch.setenv("MUTATION_REUSE", "true")
    assert main() == 1
    assert json.loads(report_path.read_text())["status"] == "failed"


@pytest.mark.parametrize("diff", ["", " "])
def test_empty_diff_cannot_be_reported_as_complete(mutation_tool: Path, diff: str) -> None:
    (mutation_tool.parent / "diffs.json").write_text(
        json.dumps({"package.x_check__mutmut_1": diff})
    )
    assert main() == 1
    assert json.loads(Path("mutation-report/report.json").read_text())["status"] == "failed"


def test_long_diff_has_bounded_preview_and_full_downloadable_output(mutation_tool: Path) -> None:
    diff = "- original\n+ " + "new content\n" * 500 + "+ final changed line\n"
    (mutation_tool.parent / "diffs.json").write_text(
        json.dumps({"package.x_check__mutmut_1": diff})
    )
    assert main() == 0
    report = json.loads(Path("mutation-report/report.json").read_text())
    preview = report["survivors"][0]["diff"]
    assert len(preview) <= 4000
    assert "diff truncated; full diff in artifact" in preview
    assert "+ final changed line" not in preview
    assert diff in Path("mutation-report/tool-output.txt").read_text()


def test_invalid_mutant_name_cannot_be_reported_as_complete(mutation_tool: Path) -> None:
    (mutation_tool.parent / "results.txt").write_text("x" * 501 + ": survived\n")
    assert main() == 1
    assert json.loads(Path("mutation-report/report.json").read_text())["status"] == "failed"
