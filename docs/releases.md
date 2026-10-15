# Documentation and releases

## Work on the docs

The Markdown source and theme customizations live in `docs/`; root `mkdocs.yml`
defines navigation and the build. Documentation dependencies live only in the
`docs` dependency group in `pyproject.toml` and are locked in `uv.lock`.
They are not runtime dependencies or a published installation extra.

```bash
uv sync --locked --group docs
uv run --group docs mkdocs serve
uv run --group docs mkdocs build --strict --site-dir /tmp/social-jev-docs
```

Set `DOCS_VERSION` for an exact release build. The canonical URL is
`https://social-jev.github.io/docs/{DOCS_VERSION}/`; local builds default to `v`
plus the version in `pyproject.toml`.

```bash
DOCS_VERSION=v0.2.0 uv run --group docs mkdocs build --strict --site-dir /tmp/social-jev-docs
```

## Publish package documentation

Pushing a `v*` tag triggers the documentation workflow. After a strict build and
confirmation that the tagged commit belongs to `main`, it dispatches:

```json
{
  "event_type": "package-docs",
  "client_payload": {"version": "v0.2.0"}
}
```

The destination is `social-jev/social-jev.github.io`. Its workflow checks out the
exact eligible package tag and builds with:

```bash
uv sync --locked --group docs
DOCS_VERSION=v0.2.0 uv run --group docs mkdocs build --strict --site-dir ABSOLUTE_OUTPUT
```

The website derives `DOCS_VERSION` from the discovered eligible tag automatically;
no manual version entry is required. The website deployment owns installation of the built files, preservation of older
versions and the landing page, and `/docs/versions.json`:

```json
{"latest": "v0.2.0", "versions": ["v0.2.0", "v0.1.0"]}
```

Initially the website publishes stable `vMAJOR.MINOR.PATCH` tags only. It skips
historical tags without `mkdocs.yml`, reconciles missing eligible versions on a
dispatch or an explicit retry, and redirects `/docs/` to latest stable.
The package selector uses that manifest, selects the current URL version, and
preserves the current page when available in the target version; otherwise it
opens that version's homepage. No publication PR or manual merge is required. There is no periodic deployment
schedule; publication is triggered by the package release notification.

### One-time credential setup

Create a fine-grained GitHub token with access only to
`social-jev/social-jev.github.io`, granting **Contents: read and write** and the
required metadata access. Add it as the package repository Actions secret
**`WEBSITE_PUBLISH_TOKEN`**. Follow the organization approval and expiry settings.
Do not reuse a broadly scoped personal token.

Until that secret is configured, the package's dispatch job fails explicitly;
configure the narrowly scoped secret and explicitly retry publication. No
credential is embedded in this repository.

## Publish the Python package

Publishing a GitHub release triggers `.github/workflows/pypi.yaml`. It checks the
release revision, verifies that the tag matches `v` plus the package version and
belongs to `main`, builds with `uv build`, and publishes through PyPI Trusted
Publishing in the `pypi` environment. This uses no PyPI API token.

For a release, update the package version and lockfile, merge and wait for checks,
then push the matching tag and publish its GitHub release. Mark alpha releases as
GitHub prereleases. Do not move published tags.

Documentation publication on tag push and PyPI publication on GitHub release are
separate workflows. Changing the project website does not require a package release.
