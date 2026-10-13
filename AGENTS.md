# AGENTS.md

## General Guidelines

### Code Style & Typing
- **Simplicity**: writing elegant, simple, maintainable, readable code is a first-class priority. Adding complexity needs strong justification.
- **Formatting**: Strict `ruff` enforcement. All PRs must pass `ruff check --fix .`
- **Typing**: Explicit types preferred
  - **OK**: `cast(...)`, `assert ...` for type narrowing
  - **SOMETIMES OK**: Untyped args for simple cases (e.g., prompt handlers)
  - **NOT OK**: blank `# type: ignore` comments without strong justification

### Naming Conventions (enforced by `ruff`)
- **Methods**: snake_case
- **Classes**: PascalCase (e.g., `LocalREPL`, `PortkeyClient`)
- **Variables**: snake_case
- **Constants**: UPPER_CASE (e.g., `SAFE_BUILTINS`, `SYSTEM_PROMPT`)
- **No leading underscore**: Do NOT use `_` prefix, including private methods, unless explicitly requested (i.e. private_method() not _private_method(), CONSTANT not _CONSTANT, Class not _Class) except to declare unused variables in e.g. loops (e.g., `for _ in range(10): ...`)

### Error Handling Philosophy
- **Fail fast, fail loud** - No defensive programming or silent fallbacks; fallbacks generally must be strongly justified by environment variation
- **Minimize branching** - Prefer single code paths; every `if`/`try` needs justification
- **Example**: Missing API key → immediate `ValueError`, not graceful fallback

### Dependencies
- Avoid new core dependencies
- Use optional extras for non-essential features (e.g., `modal` extra)
- Exception: tiny deps that simplify widely-used code

### Testing
- `uv run pytest` with discovery under `tests/`
- Write simple, deterministic unit tests
- Update tests when changing functionality
- For isolated environments, mock external services

### Documentation
- Keep concise and actionable
- Update README when behavior changes
- Avoid content duplication

### Scope

- **Do not** aim for backward compatibility
- **Do not** add any additional fluff, functionality, or features beyond the scope of the task at hand
- **Do not** add new dependencies unless strongly justified
- **Do not** touch files unrelated to the task at hand
- **Do not** fix unrelated discovered errors you might come across -- report them

- **Do** strive for the most minimal, focussed, and maintainable solution
- **Do** aim for small, focussed diffs per commit and PR
- **Do** delete dead code (don't guard it)
- **Do** aim for one well-defined change per PR

### Commits
- While executing a task, regularly commit completed subtask edits so changes can be cherry-picked granularly.
- Each commit must have a message and a body.
- Use commit messages in the form `[<type of task>] <intuitive commit>`, e.g. `[refactor] moved example.py to repo-root`.
- Both the commit subject and commit body must be plain and intuitive to someone who does not know this project. The subject should state the practical change; the body should explain any necessary names, before/after behavior, and why the change matters.

### Checklist

Before completing a PR, run the same checks as CI:

```bash
uv run prek run --all-files
```

Vulture is part of the normal checks. For production, test, dependency, or CI changes, also run mutation testing explicitly:

```bash
uv run prek run mutmut --stage manual --all-files
```

The manual stage keeps mutation testing out of ordinary local commits. In CI, request it explicitly by commenting `/mutmut` on the PR. It runs only after normal checks pass for that version; new pushes require another request. Report surviving mutations separately from execution failures; do not equate a successful command with complete test protection.
