# Publishing a release

Publishing a GitHub release triggers `.github/workflows/pypi.yaml`:

1. Run the repository's normal checks on the release revision.
2. Confirm the tagged commit belongs to `main` and its tag is `v` plus the
   version in `pyproject.toml`.
3. Build the wheel and source distribution with `uv build`.
4. Publish those artifacts through PyPI Trusted Publishing in the `pypi`
   GitHub environment.

Configure the pending publisher in the PyPI account's Publishing settings:

| Field | Value |
| --- | --- |
| PyPI project | `social-jev` |
| GitHub owner | `DGoettlich` |
| Repository | `social-jev` |
| Workflow | `pypi.yaml` |
| Environment | `pypi` |

No PyPI API token is needed. The first successful upload creates the project;
configuring the pending publisher does not reserve the name.

For each release, update the version, merge the change and wait for checks to
pass, then create the matching tag and publish its GitHub release. Mark alpha
versions as GitHub prereleases. The tagged revision must include the publishing
workflow. Do not move published release tags.

The first alpha tag predates this workflow. After merging the workflow, use its
manual Run workflow entry on `main` with release tag `v0.1.0a1` to publish that
existing revision. The same checks and publishing job run, without moving the
tag. The manual trigger also supports retrying releases that predate the workflow.
