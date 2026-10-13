"""Dispatch explicit PR mutation requests and reject stale worker runs."""

import json
import os
import subprocess
from pathlib import Path


def api(endpoint: str, **fields: object):
    command = ["gh", "api", f"repos/{os.environ['GITHUB_REPOSITORY']}/{endpoint}"]
    if fields:
        command.extend(["--input", "-"])
    try:
        result = subprocess.run(
            command,
            input=json.dumps(fields),
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        message = "GitHub request failed. Check Actions for an existing run before retrying."
        raise ValueError(message) from None
    return json.loads(result.stdout) if result.stdout else None


def dispatch(event: dict) -> None:
    if (
        event["action"] != "created"
        or event["comment"]["body"].strip() != "/mutmut"
        or "pull_request" not in event["issue"]
    ):
        return
    permission = api(f"collaborators/{event['comment']['user']['login']}/permission")
    if permission["permission"] not in {"admin", "maintain", "write"}:
        message = "Mutation requests require repository write access."
        raise ValueError(message)
    pr = pull_request(str(event["issue"]["number"]))
    api(
        "actions/workflows/ci.yaml/dispatches",
        ref=pr["head"]["ref"],
        inputs={
            "mutation_pr": str(event["issue"]["number"]),
            "mutation_revision": pr["head"]["sha"],
        },
    )
    api(f"issues/comments/{event['comment']['id']}/reactions", content="eyes")


def pull_request(number: str) -> dict:
    if not number.isdecimal() or int(number) < 1:
        message = "A positive PR number is required."
        raise ValueError(message)
    pr = api(f"pulls/{number}")
    repository = pr["head"]["repo"]
    if (
        pr["state"] != "open"
        or repository is None
        or repository["full_name"] != os.environ["GITHUB_REPOSITORY"]
    ):
        message = "Mutation requests require an open PR from this repository."
        raise ValueError(message)
    return pr


def validate(event: dict) -> None:
    pr = pull_request(event["inputs"]["mutation_pr"])
    if not pr["head"]["sha"] == event["inputs"]["mutation_revision"] == os.environ["GITHUB_SHA"]:
        message = "The PR changed after this request. Comment /mutmut again."
        raise ValueError(message)


def main() -> int:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    try:
        if os.environ["GITHUB_EVENT_NAME"] == "issue_comment":
            dispatch(event)
        else:
            validate(event)
    except ValueError as error:
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as output:
            output.write(f"Mutation request failed: {error}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
