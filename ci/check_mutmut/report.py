"""Collect mutation results without confusing successful execution with test strength."""

import html
import json
import os
import subprocess
from pathlib import Path
from typing import TypedDict, cast

MUTATION_PATHS = [
    "src/**",
    "ci/**",
    "tests/**",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "prek.toml",
    ".github/workflows/ci.yaml",
    ".github/workflows/mutmut.yaml",
]

# these fields come from mutmut export-cicd-stats; omitted results are counted separately
MUTMUT_COUNT_FIELDS = {
    "total",
    "killed",
    "survived",
    "no_tests",
    "skipped",
    "suspicious",
    "timeout",
    "check_was_interrupted_by_user",
    "segfault",
}
REPORT_VERSION = 2
MAX_SURVIVORS = 20
MAX_DESCRIPTION = 500
MAX_DIFF = 4000
DIFF_TRUNCATION = "\n… [diff truncated; full diff in artifact]\n"
INVALID_DATA = "Mutation results are missing or inconsistent."


class Survivor(TypedDict):
    name: str
    diff: str


class Report(TypedDict):
    version: int
    status: str
    reused: bool
    producer: str
    counts: dict[str, int] | None
    survivors: list[Survivor]


def collect(report: Report, directory: Path) -> None:
    raw = Path("mutants/mutmut-cicd-stats.json")
    raw.unlink(missing_ok=True)
    with (directory / "tool-output.txt").open("w") as output:
        subprocess.run(
            ["mutmut", "export-cicd-stats"], stdout=output, stderr=output, check=True, timeout=60
        )
        result = subprocess.run(
            ["mutmut", "results"], capture_output=True, text=True, check=False, timeout=60
        )
        output.write(result.stdout + result.stderr)
        result.check_returncode()
    counts = json.loads(raw.read_text())
    validate_counts(counts, MUTMUT_COUNT_FIELDS)
    counts["not_checked"] = counts["total"] - sum(
        value for key, value in counts.items() if key != "total"
    )
    if counts["not_checked"] < 0:
        raise ValueError(INVALID_DATA)
    report["counts"] = counts
    names = [
        line.strip().removesuffix(": survived")
        for line in result.stdout.splitlines()
        if line.strip().endswith(": survived")
    ][:MAX_SURVIVORS]
    report["survivors"] = [survivor(name, directory) for name in names]
    validate_survivors(report["survivors"], counts["survived"])
    report["status"] = status(counts, os.environ.get("MUTATION_OUTCOME", "failure"))


def survivor(name: str, directory: Path) -> Survivor:
    if not name.strip() or len(name) > MAX_DESCRIPTION:
        raise ValueError(INVALID_DATA)
    result = subprocess.run(
        ["mutmut", "show", name], capture_output=True, text=True, check=False, timeout=60
    )
    with (directory / "tool-output.txt").open("a") as output:
        output.write(result.stdout + result.stderr)
    result.check_returncode()
    diff = result.stdout
    if len(diff) > MAX_DIFF:
        diff = diff[: MAX_DIFF - len(DIFF_TRUNCATION)] + DIFF_TRUNCATION
    return {"name": name, "diff": diff}


def status(counts: dict[str, int], outcome: str) -> str:
    if counts["not_checked"] or counts["check_was_interrupted_by_user"] or outcome == "cancelled":
        return "incomplete"
    if outcome != "success" or any(counts[key] for key in ("timeout", "suspicious", "segfault")):
        return "failed"
    return "complete"


def validate_counts(counts: object, fields: set[str]) -> None:
    if not isinstance(counts, dict) or counts.keys() != fields:
        raise ValueError(INVALID_DATA)
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError(INVALID_DATA)


def reuse(directory: Path, producer: str) -> Report:
    report = json.loads((directory / "report.json").read_text())
    if (
        not isinstance(report, dict)
        or type(report.get("version")) is not int
        or report["version"] != REPORT_VERSION
        or report.get("status") != "complete"
    ):
        raise ValueError(INVALID_DATA)
    validate_counts(report.get("counts"), MUTMUT_COUNT_FIELDS | {"not_checked"})
    counts = report["counts"]
    if counts["total"] != sum(value for key, value in counts.items() if key != "total"):
        raise ValueError(INVALID_DATA)
    if status(counts, "success") != "complete":
        raise ValueError(INVALID_DATA)
    validate_details(report, producer)
    report["reused"] = True
    return cast("Report", report)


def validate_details(report: dict, producer: str) -> None:
    origin = producer.rsplit("/", 1)[0] + "/"
    link = report.get("producer", "")
    if (
        not isinstance(link, str)
        or not link.startswith(origin)
        or not link[len(origin) :].isdigit()
    ):
        raise ValueError(INVALID_DATA)
    validate_survivors(report.get("survivors"), report["counts"]["survived"])


def validate_survivors(survivors: object, expected: int) -> None:
    if not isinstance(survivors, list) or len(survivors) != min(expected, MAX_SURVIVORS):
        raise ValueError(INVALID_DATA)
    if any(not isinstance(item, dict) or item.keys() != {"name", "diff"} for item in survivors):
        raise ValueError(INVALID_DATA)
    if any(
        not isinstance(item[key], str) or not item[key].strip() or len(item[key]) > limit
        for item in survivors
        for key, limit in (("name", MAX_DESCRIPTION), ("diff", MAX_DIFF))
    ):
        raise ValueError(INVALID_DATA)


def markdown(report: Report) -> str:
    counts = report["counts"]
    details = (
        "Counts unavailable.\n"
        if counts is None
        else "\n".join(f"- {key.replace('_', ' ')}: {value}" for key, value in counts.items())
    )
    survivors = "\n".join(
        "<details><summary>"
        + html.escape(
            item["name"].replace(".x_", ".").replace("__mutmut_", " · mutation ")
        ).replace("@", "&#64;")
        + "</summary>\n\n<pre>"
        + html.escape(item["diff"]).replace("@", "&#64;")
        + "</pre>\n</details>"
        for item in report["survivors"]
    )
    reuse = " (reused result)" if report["reused"] else ""
    return (
        f"## Mutation testing — {report['status']}{reuse}\n\n{details}\n\n"
        f"{survivors}\n\n[Producing run]({report['producer']})\n"
    )


def main() -> int:
    directory = Path("mutation-report")
    directory.mkdir(exist_ok=True)
    report: Report = {
        "version": REPORT_VERSION,
        "status": "failed",
        "reused": False,
        "producer": (
            f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
            f"{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/"
            f"{os.environ.get('GITHUB_RUN_ID', '')}"
        ),
        "counts": None,
        "survivors": [],
    }
    try:
        if os.environ.get("MUTATION_REUSE") == "true":
            report = reuse(directory, report["producer"])
        else:
            collect(report, directory)
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        report["status"] = "failed"
        report["counts"] = None
        report["survivors"] = []
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (directory / "summary.md").write_text(markdown(report))
    return int(report["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
