"""Report bounded test-quality findings without importing test modules."""

import ast
import html
import io
import os
import re
import sys
import tokenize
import tomllib
from pathlib import Path

COMMENT_SECTION_LIMIT = 20000

MESSAGES = {
    "TST001": "test only checks default field values; remove this test.",
    "TST002": (
        "assertion checks a truthy literal and always passes; "
        "assert an observable result of the code under test or remove the assertion."
    ),
    "TST003": (
        "assertion pins a signature default; test behavior with the argument omitted "
        "or explain the signature's public contract in an inline exception."
    ),
    "TST004": (
        "assertion appears to repeat a mock's configured return value; "
        "test how production code uses that value or remove the test."
    ),
    "TST005": (
        "assertion appears to pin documentation text or version metadata; test the behavior "
        "it supports or explain why the exact value is a public contract in an inline exception."
    ),
}


def literal(node: ast.expr) -> str:
    try:
        return "literal:" + repr(ast.literal_eval(node))
    except (ValueError, TypeError):
        return ""


def file_kind(node: ast.expr, bindings: dict[str, str]) -> str:
    name = value(node, bindings).removeprefix("literal:")
    if not name.startswith(("'", '"')):
        return ""
    path = Path(ast.literal_eval(name))
    documentation = {"docs", "documentation"}.intersection(part.lower() for part in path.parts)
    conventional = path.stem.lower() in {"readme", "changelog", "contributing", "agents", "skill"}
    if path.suffix.lower() in {".md", ".rst"} and (documentation or conventional):
        return "documentation"
    return "manifest" if path.name in {"pyproject.toml", "package.json"} else ""


def call_value(node: ast.Call, bindings: dict[str, str]) -> str:
    called = value(node.func, bindings)
    if called in {"inspect.signature", "importlib.metadata.version"}:
        return {"inspect.signature": "signature", "importlib.metadata.version": "metadata"}[called]
    if called in {"unittest.mock.Mock", "unittest.mock.MagicMock"} and not any(
        k.arg == "side_effect" for k in node.keywords
    ):
        configured = next(
            (value(k.value, bindings) for k in node.keywords if k.arg == "return_value"), ""
        )
        return "mock:" + configured
    if called.startswith("mock:"):
        return "echo:" + called.removeprefix("mock:")
    return data_call(node, called, bindings)


def data_call(node: ast.Call, called: str, bindings: dict[str, str]) -> str:
    if called in {"pathlib.Path", "open"} and node.args:
        prefix = "path:" if called == "pathlib.Path" else "file:"
        return prefix + file_kind(node.args[0], bindings)
    if called.startswith("path:") and called.endswith(".open"):
        return called.replace("path:", "file:").removesuffix(".open")
    if called.startswith(("path:", "file:")) and called.endswith((".read_text", ".read")):
        return called.split(":")[1].split(".", maxsplit=1)[0]
    if called in {"json.loads", "tomllib.loads", "json.load", "tomllib.load"} and node.args:
        return (
            "manifest-data"
            if value(node.args[0], bindings) in {"manifest", "file:manifest"}
            else ""
        )
    name = getattr(node.func, "id", getattr(node.func, "attr", ""))
    return "object" if name[:1].isupper() and not node.args and not node.keywords else ""


def attribute_value(node: ast.Attribute, bindings: dict[str, str]) -> str:
    if node.attr in {"__defaults__", "__kwdefaults__"}:
        return "default"
    if node.attr in {"__doc__", "__version__"}:
        return "metadata"
    parent = value(node.value, bindings)
    return {
        ("object", node.attr): "field",
        ("signature", "parameters"): "parameters",
        ("parameter", "default"): "default",
    }.get((parent, node.attr), f"{parent}.{node.attr}")


def subscript_value(node: ast.Subscript, bindings: dict[str, str]) -> str:
    parent = value(node.value, bindings)
    if parent == "manifest-data":
        return "metadata" if literal(node.slice) == "literal:'version'" else parent
    return {"parameters": "parameter", "default": "default"}.get(parent, "")


def value(node: ast.expr, bindings: dict[str, str]) -> str:
    match node:
        case ast.Name(id=name):
            bound = bindings.get(name, "")
            return "metadata" if bound.endswith((".__version__", ".__doc__")) else bound
        case ast.Call():
            return call_value(node, bindings)
        case ast.Attribute():
            return attribute_value(node, bindings)
        case ast.Subscript():
            return subscript_value(node, bindings)
    return literal(node)


def bind_target(target: ast.expr, kind: str, bindings: dict[str, str]) -> None:
    if isinstance(target, ast.Name):
        bindings[target.id] = kind
        return
    if isinstance(target, ast.Attribute) and target.attr == "return_value":
        original = value(target.value, bindings)
        invalidate_mocks(bindings)
        if original.startswith("mock:") and isinstance(target.value, ast.Name):
            bindings[target.value.id] = "mock:" + kind
        return
    for node in ast.walk(target):
        if isinstance(node, ast.Name):
            bindings[node.id] = ""
    invalidate_mocks(bindings)


def bind(statement: ast.stmt, bindings: dict[str, str]) -> bool:
    match statement:
        case ast.Import(names=names):
            for name in names:
                bindings[name.asname or name.name.split(".")[0]] = (
                    name.name if name.asname else name.name.split(".")[0]
                )
        case ast.ImportFrom(module=module, names=names):
            bindings.update({name.asname or name.name: f"{module}.{name.name}" for name in names})
        case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name) | ast.ClassDef(name=name):
            bindings[name] = ""
            invalidate_mocks(bindings)
        case ast.Assign(targets=targets, value=expression):
            assign(targets, expression, bindings)
        case _:
            return False
    return True


def assign(targets: list[ast.expr], expression: ast.expr, bindings: dict[str, str]) -> None:
    kind = value(expression, bindings)
    if not kind.startswith(("mock:", "echo:")) and any(
        isinstance(n, ast.Call) for n in ast.walk(expression)
    ):
        invalidate_mocks(bindings)
    for target in targets:
        bind_target(target, kind, bindings)


def with_scan(node: ast.With | ast.AsyncWith, bindings: dict[str, str]) -> list[tuple[int, str]]:
    local = bindings.copy()
    invalidate_mocks(local)
    for item in node.items:
        if item.optional_vars:
            bind_target(item.optional_vars, value(item.context_expr, local), local)
    return scan(node.body, local)


def nested_scan(node: ast.AST) -> list[tuple[int, str]]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return []
    result = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.stmt):
            result.extend(scan([child], {}))
        elif isinstance(child, (ast.ExceptHandler, ast.match_case)):
            result.extend(scan(child.body, {}))
    return result


def invalidate_mocks(bindings: dict[str, str]) -> None:
    for name, kind in list(bindings.items()):
        if kind.startswith(("mock:", "echo:")):
            bindings[name] = ""


def comparison_rules(node: ast.Compare, bindings: dict[str, str]) -> list[str]:
    if len(node.ops) != 1 or not isinstance(node.ops[0], (ast.Eq, ast.Is, ast.In)):
        return []
    kinds = [value(node.left, bindings), value(node.comparators[0], bindings)]
    result = []
    if any(k.startswith("literal:") for k in kinds):
        if "default" in kinds:
            result.append("TST003")
        if {"metadata", "documentation"}.intersection(kinds):
            result.append("TST005")
    if isinstance(node.ops[0], (ast.Eq, ast.Is)) and any(
        k.startswith("echo:literal:") and k.removeprefix("echo:") in kinds for k in kinds
    ):
        result.append("TST004")
    return result


def assertion_rules(node: ast.Assert, bindings: dict[str, str]) -> list[tuple[int, str]]:
    result = []
    if isinstance(node.test, ast.Constant) and node.test.value:
        result.append((node.lineno, "TST002"))
    if any(
        isinstance(n, ast.Call) and not value(n, bindings).startswith("echo:")
        for n in ast.walk(node.test)
    ):
        invalidate_mocks(bindings)
    for expression in ast.walk(node.test):
        if isinstance(expression, ast.Compare):
            result.extend((node.lineno, rule) for rule in comparison_rules(expression, bindings))
    return result


def scan(body: list[ast.stmt], bindings: dict[str, str]) -> list[tuple[int, str]]:
    result = []
    for statement in body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue
        if bind(statement, bindings):
            continue
        if isinstance(statement, ast.Assert):
            result.extend(assertion_rules(statement, bindings))
        else:
            # branch-local facts cannot establish what happened on another path
            result.extend(
                with_scan(statement, bindings)
                if isinstance(statement, (ast.With, ast.AsyncWith))
                else nested_scan(statement)
            )
            bindings.clear()
    return result


def default_only(body: list[ast.stmt]) -> list[tuple[int, str]]:
    bindings: dict[str, str] = {}
    result = []
    for statement in body:
        match statement:
            case ast.Assign(targets=[ast.Name(id=name)], value=expression):
                kind = value(expression, bindings)
                bindings[name] = kind
                if kind != "object":
                    break
            case ast.Assert(
                test=ast.Compare(left=left, ops=[ast.Eq() | ast.Is()], comparators=[right])
            ):
                if {value(left, bindings).split(":")[0], value(right, bindings).split(":")[0]} != {
                    "field",
                    "literal",
                }:
                    return []
                result.append((statement.lineno, "TST001"))
            case _:
                break
    else:
        return result
    return []


def findings(tree: ast.Module) -> list[tuple[int, str]]:
    result = []
    imports: dict[str, str] = {"open": "open"}
    for statement in tree.body:
        if not bind(statement, imports) and not (
            isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
        ):
            imports.clear()
    candidates = list(tree.body)
    for function in candidates:
        if isinstance(function, ast.ClassDef) and function.name.startswith("Test"):
            candidates.extend(function.body)
        if isinstance(
            function, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and function.name.startswith("test_"):
            arguments = {arg.arg for arg in ast.walk(function.args) if isinstance(arg, ast.arg)}
            bindings = {name: kind for name, kind in imports.items() if name not in arguments}
            result.extend(
                default_only(
                    [
                        s
                        for s in function.body
                        if not isinstance(s, ast.Expr) or not isinstance(s.value, ast.Constant)
                    ]
                )
            )
            result.extend(scan(function.body, bindings))
    return sorted(set(result))


def suppressions(source: str) -> tuple[set[tuple[int, str]], list[tuple[int, str, str]]]:
    ignored: set[tuple[int, str]] = set()
    errors = []
    for comment in tokenize.generate_tokens(io.StringIO(source).readline):
        if comment.type != tokenize.COMMENT or "test-quality:" not in comment.string:
            continue
        match = re.fullmatch(r"#\s*test-quality: ignore (TST\d{3}) - (\S.*)", comment.string)
        if match and match[1] in MESSAGES:
            ignored.add((comment.start[0], match[1]))
        else:
            errors.append((comment.start[0], "error", "invalid test-quality suppression"))
    return ignored, errors


def check_file(path: Path) -> list[tuple[int, str, str]]:
    try:
        with tokenize.open(path) as stream:
            source = stream.read()
        tree = ast.parse(source, filename=str(path))
        ignored, errors = suppressions(source)
    except (OSError, UnicodeError) as error:
        return [(1, "error", f"cannot read input: {error}")]
    except (SyntaxError, tokenize.TokenError) as error:
        return [(getattr(error, "lineno", 1) or 1, "error", f"invalid Python input: {error}")]
    return sorted(
        errors
        + [
            (line, rule, MESSAGES[rule])
            for line, rule in findings(tree)
            if (line, rule) not in ignored
        ]
    )


def warning_rules(path: Path) -> set[str]:
    try:
        with path.open("rb") as stream:
            tool = tomllib.load(stream).get("tool", {})
    except FileNotFoundError:
        return set()
    if not isinstance(tool, dict):
        message = "tool must be a table"
        raise TypeError(message)
    return validate_rules(tool.get("check-tests", {}))


def validate_rules(config: object) -> set[str]:
    if not isinstance(config, dict) or config.keys() - {"fail", "warn"}:
        message = "tool.check-tests must be a table containing only fail and warn lists"
        raise ValueError(message)
    for level, rules in config.items():
        if not isinstance(rules, list) or any(
            not isinstance(rule, str) or rule not in MESSAGES for rule in rules
        ):
            message = f"{level} must be a list of known rule IDs: {', '.join(MESSAGES)}"
            raise ValueError(message)
    warn = set(config.get("warn", []))
    if warn.intersection(config.get("fail", [])):
        message = "rule IDs must not appear in both fail and warn"
        raise ValueError(message)
    return warn


def escape(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def report(path: Path, line: int, rule: str, message: str, warn: set[str]) -> bool:
    # unlisted rules fail, so new checks cannot silently become nonblocking
    blocking = rule not in warn
    if os.environ.get("GITHUB_ACTIONS") == "true":
        # the runner parses both modern and legacy commands in plain log lines
        filename = escape(str(path)).replace(":", "%3A").replace(",", "%2C")
        level = "error" if blocking else "warning"
        print(f"::{level} file={filename},line={line}::{escape(rule + ': ' + message)}")
    else:
        diagnostic = f"{path}:{line}: {rule}: {message}"
        print(diagnostic.replace("\r", r"\r").replace("\n", r"\n"))
    return blocking


def summary(findings: list[tuple[Path, int, str, str]], warn: set[str]) -> str:
    groups: dict[str, list[str]] = {"Failures": [], "Warnings": []}
    for path, line, rule, message in findings:
        group = "Warnings" if rule in warn else "Failures"
        groups[group].append(html.escape(f"{path}:{line}: {rule}: {message}").replace("@", "&#64;"))
    state = "🔴 Failed" if groups["Failures"] else "🟢 Completed"
    sections = [f"## 🔎 Test quality\n\n{state}\n"]
    for name, entries in groups.items():
        # leave room for both groups and publishing metadata within github's comment limit
        lines = []
        size = 0
        for entry in entries:
            if size + len(entry) > COMMENT_SECTION_LIMIT:
                lines.append("… remaining findings are in the run logs")
                break
            lines.append(entry)
            size += len(entry) + 1
        body = "<pre>" + "\n".join(lines) + "</pre>\n" if lines else ""
        sections.append(
            f"<details>\n<summary>{name} ({len(entries)})</summary>\n\n{body}\n</details>\n"
        )
    return "\n".join(sections)


def check_paths(paths: list[str]) -> list[tuple[Path, int, str, str]]:
    files: set[Path] = set()
    for filename in paths:
        path = Path(filename)
        files.update(path.rglob("*.py") if path.is_dir() else [path])
    return [
        (path, line, rule, message)
        for path in sorted(files)
        for line, rule, message in check_file(path)
    ]


def main(paths: list[str] | None = None) -> int:
    config = Path("pyproject.toml")
    warn: set[str] = set()
    try:
        warn = warning_rules(config)
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        findings = [(config, 1, "error", f"invalid configuration: {error}")]
    else:
        findings = check_paths(paths if paths is not None else sys.argv[1:])
    failed = False
    for path, line, rule, message in findings:
        failed |= report(path, line, rule, message, warn)
    if output := os.environ.get("TEST_QUALITY_REPORT"):
        Path(output).write_text(summary(findings, warn), encoding="utf-8")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
