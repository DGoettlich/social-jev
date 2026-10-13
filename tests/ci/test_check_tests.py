"""Exercise test-quality diagnostics through the command boundary."""

from pathlib import Path

import pytest

from ci.check_tests import main


@pytest.fixture(autouse=True)
def local_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # output-mode tests opt into annotations; other examples must not inherit the host's mode
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.chdir(tmp_path)


def test_literal_passing_assertion_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "test_sample.py"
    path.write_text('raise RuntimeError("must not execute")\ndef test_bad():\n    assert True\n')
    assert main([str(path)]) == 1
    assert f"{path}:3: TST002:" in capsys.readouterr().out


@pytest.mark.parametrize(
    "body",
    [
        "item = Config()\nassert item.timeout == 30",
        "assert Config().timeout == 30",
        "item = Config()\nother = item\nassert other.timeout == 30",
    ],
)
def test_default_only_tests_fail(
    body: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "test_sample.py"
    path.write_text("async def test_defaults():\n    " + body.replace("\n", "\n    ") + "\n")
    assert main([str(path)]) == 1
    assert "TST001:" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("setup", "expression"),
    [
        ("import inspect", 'inspect.signature(target).parameters["timeout"].default'),
        ("from inspect import signature as sig", 'sig(target).parameters["timeout"].default'),
        (
            "import inspect as ins\ns = ins.signature(target)\np = s.parameters['timeout']",
            "p.default",
        ),
        ("", "target.__defaults__[0]"),
        ("", "target.__kwdefaults__['timeout']"),
    ],
)
def test_signature_defaults_fail(
    setup: str, expression: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "test_sample.py"
    body = setup + f"\nassert {expression} == 30"
    path.write_text("def test_signature():\n    " + body.strip().replace("\n", "\n    ") + "\n")
    assert main([str(path)]) == 1
    assert "TST003:" in capsys.readouterr().out


@pytest.mark.parametrize(
    "body",
    [
        "from unittest.mock import Mock\nm = Mock(return_value=3)\nassert m() == 3",
        (
            "from unittest.mock import MagicMock as Fake\n"
            "m = Fake()\n"
            "m.return_value = 3\n"
            "result = m()\n"
            "assert result == 3"
        ),
        (
            "import unittest.mock as mock\n"
            "m = mock.Mock(return_value=3)\n"
            "alias = m\n"
            "assert alias() == 3"
        ),
    ],
)
def test_mock_return_echo_fail(
    body: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "test_sample.py"
    path.write_text("def test_mock():\n    " + body.replace("\n", "\n    ") + "\n")
    assert main([str(path)]) == 1
    assert "TST004:" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("setup", "expression"),
    [
        ("", "target.__doc__ == 'hello'"),
        ("", "package.__version__ == '1.0'"),
        ("from importlib.metadata import version as installed", "installed('pkg') == '1.0'"),
        ("from pathlib import Path\ndoc = Path('README.md').read_text()", "'hello' in doc"),
        (
            "import pathlib as p\nf = p.Path('docs/intro.rst')\ndoc = f.read_text()",
            "doc == 'hello'",
        ),
        (
            (
                "import json\n"
                "from pathlib import Path\n"
                "data = json.loads(Path('package.json').read_text())"
            ),
            "data['version'] == '1.0'",
        ),
        (
            (
                "import tomllib\n"
                "from pathlib import Path\n"
                "data = tomllib.loads(Path('pyproject.toml').read_text())"
            ),
            "data['project']['version'] == '1.0'",
        ),
    ],
)
def test_metadata_pinning_fail(
    setup: str, expression: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "test_sample.py"
    body = setup + f"\nassert {expression}"
    path.write_text("def test_metadata():\n    " + body.strip().replace("\n", "\n    ") + "\n")
    assert main([str(path)]) == 1
    assert "TST005:" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("source", "code"),
    [
        (
            (
                "def test_good():\n"
                "    assert True  # test-quality: ignore TST002 - deliberate smoke contract\n"
            ),
            0,
        ),
        ("def test_bad():\n    assert True  # test-quality: ignore TST002\n", 1),
        ("def test_bad():\n    pass  # test-quality: ignore TST999 - typo\n", 1),
        ("def test_bad(:\n", 1),
    ],
)
def test_invalid_inputs_and_suppressions(
    source: str, code: int, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "test_sample.py"
    path.write_text(source)
    assert main([str(path)]) == code
    output = capsys.readouterr().out
    if code:
        assert f"{path}:" in output
    else:
        assert output == ""


def test_directory_scan_and_safe_annotations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    path = tmp_path / "test_,%\n::error::injected.py"
    path.write_text("def test_bad():\n    assert True\n")
    (tmp_path / "ignored.txt").write_text("invalid python")
    assert main([str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert "%2C%25%0A%3A%3Aerror%3A%3Ainjected.py,line=2" in output
    assert "\n::error::injected" not in output


def test_missing_input_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(tmp_path / "missing.py")]) == 1
    assert "cannot read" in capsys.readouterr().out


@pytest.mark.parametrize(
    "body",
    [
        "config = Config()\nconfig.fetch()\nassert config.timeout == 30",
        "config = Config(timeout=30)\nassert config.timeout == 30",
        "assert fetch().status == 200",
        (
            "import inspect\n"
            "sig = inspect.signature\n"
            "sig = custom\n"
            "assert sig(target).parameters['x'].default == 3"
        ),
        "from inspect import signature\nassert signature(target).value == 3",
        (
            "from unittest.mock import Mock\n"
            "m = Mock(return_value=3)\n"
            "result = production(m)\n"
            "assert result == 3"
        ),
        (
            "from unittest.mock import Mock\n"
            "m = Mock(return_value=3)\n"
            "result = production(m) + 1\n"
            "assert m() == 3"
        ),
        (
            "from unittest.mock import Mock\n"
            "m = Mock(return_value=3)\n"
            "m.side_effect = ValueError\n"
            "assert m() == 3"
        ),
        "from unittest.mock import Mock\nm = Mock(return_value=3)\nm = production\nassert m() == 3",
        (
            "import json\n"
            "from pathlib import Path\n"
            "data = json.loads(Path('fixture.json').read_text())\n"
            "assert data['version'] == '1.0'"
        ),
        "data = {'version': '1.0'}\nassert data['version'] == '1.0'",
        (
            "from pathlib import Path\n"
            "summary = Path('mutation-report/summary.md').read_text()\n"
            "assert 'killed' in summary"
        ),
        "assert result == 3",
        "assert False",
        "assert 0",
        "source = 'assert True # test-quality: ignore TST999'\nassert parse(source).valid",
        "def helper():\n    assert True\nassert validate(helper)",
    ],
)
def test_behavioral_tests_and_unknown_provenance_pass(
    body: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "test_sample.py"
    path.write_text("def test_behavior():\n    " + body.replace("\n", "\n    ") + "\n")
    assert main([str(path)]) == 0
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "source",
    [
        "def test_branch():\n    if enabled:\n        assert True\n",
        (
            "def test_exception():\n"
            "    try:\n"
            "        run()\n"
            "    except ValueError:\n"
            "        assert 1\n"
        ),
        "class TestCase:\n    async def test_case(self):\n        assert 'ok'\n",
    ],
)
def test_literal_assertions_in_nested_test_blocks_fail(
    source: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "test_sample.py"
    path.write_text(source)
    assert main([str(path)]) == 1
    assert "TST002:" in capsys.readouterr().out


def test_import_alias_parameter_shadowing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "test_sample.py"
    path.write_text(
        "from inspect import signature\ndef test_case(signature):\n"
        "    assert signature(target).parameters['x'].default == 3\n"
    )
    assert main([str(path)]) == 0
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "body",
    [
        (
            "from pathlib import Path\n"
            "import json\n"
            "assert json.load(Path('package.json').open())['version'] == '1'"
        ),
        (
            "import tomllib\n"
            "with open('pyproject.toml', 'rb') as stream:\n"
            "    data = tomllib.load(stream)\n"
            "    assert data['project']['version'] == '1'"
        ),
        "with open('README.md') as stream:\n    text = stream.read()\n    assert 'hello' in text",
        "from package import __version__\nassert __version__ == '1'",
    ],
)
def test_metadata_file_streams_and_imports_fail(
    body: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "test_sample.py"
    path.write_text(
        '"""metadata contracts"""\n'
        + "def test_metadata():\n    "
        + body.replace("\n", "\n    ")
        + "\n"
    )
    assert main([str(path)]) == 1
    assert "TST005:" in capsys.readouterr().out


@pytest.mark.parametrize(
    "body",
    [
        (
            "from unittest.mock import Mock\n"
            "m = Mock(return_value=3, side_effect=fail)\nassert m() == 3"
        ),
        (
            "from unittest.mock import Mock\n"
            "m = Mock(return_value=3)\n"
            "assert production(m)\n"
            "assert m() == 3"
        ),
        (
            "if condition:\n"
            "    import inspect\n"
            "else:\n"
            "    assert inspect.signature(target).parameters['x'].default == 3"
        ),
        "import inspect\nassert inspect.signature(target).parameters['x'].default > 0",
    ],
)
def test_effects_and_separate_branches_do_not_leak_facts(
    body: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "test_sample.py"
    path.write_text("def test_behavior():\n    " + body.replace("\n", "\n    ") + "\n")
    assert main([str(path)]) == 0
    assert capsys.readouterr().out == ""


def test_nested_uncollected_test_is_not_scanned(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "test_sample.py"
    path.write_text("def helper():\n    def test_unused():\n        assert True\n")
    assert main([str(path)]) == 0
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("github_actions", ["false", "true"])
def test_diagnostics_are_ordered_and_suppression_is_specific(
    github_actions: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", github_actions)
    second = tmp_path / "test_b.py"
    first = tmp_path / "test_a.py"
    second.write_text("def test_b():\n    assert True\n")
    first.write_text(
        "def test_a():\n    assert True  # test-quality: ignore TST001 - separate rule\n"
    )
    assert main([str(second), str(first), str(second)]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    prefix = "::error file=" if github_actions == "true" else ""
    location = ",line=2::TST002:" if github_actions == "true" else ":2: TST002:"
    assert lines[0].startswith(f"{prefix}{first}{location}")
    assert lines[1].startswith(f"{prefix}{second}{location}")


@pytest.mark.parametrize(
    "filename", ["::error::injected.py", " ::error::injected.py", "##[error]injected.py"]
)
def test_relative_filename_cannot_inject_a_workflow_command(
    filename: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    path = Path(filename)
    path.write_text('def test_version():\n    assert package.__version__ == "1.0"\n')
    assert main([str(path)]) == 1
    output = capsys.readouterr().out.splitlines()
    assert len(output) == 1
    assert output[0].startswith("::error file=")
    assert ",line=2::TST005:" in output[0]


@pytest.mark.parametrize(
    "definition",
    [
        "def signature(target):\n    return target.describe()",
        "async def signature(target):\n    return target.describe()",
        "class signature:\n    pass",
    ],
)
@pytest.mark.parametrize("local", [False, True])
def test_definitions_shadow_imported_names(
    definition: str, local: bool, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "test_sample.py"
    check = "assert signature(target).parameters['x'].default == 3"
    body = f"{definition}\n{check}" if local else check
    prefix = "from inspect import signature\n" + ("" if local else definition + "\n")
    path.write_text(prefix + "def test_behavior():\n    " + body.replace("\n", "\n    ") + "\n")
    assert main([str(path)]) == 0
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("context", ["with production(m):", "async with production(m):"])
def test_context_entry_invalidates_mock_facts(
    context: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "test_sample.py"
    path.write_text(
        "from unittest.mock import Mock\nasync def test_behavior():\n"
        "    m = Mock(return_value=3)\n"
        f"    {context}\n"
        "        assert m() == 3\n"
    )
    assert main([str(path)]) == 0
    assert capsys.readouterr().out == ""


def test_module_control_flow_invalidates_import_facts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "test_sample.py"
    path.write_text(
        "from inspect import signature\nif flag:\n    signature = custom\n"
        "def test_behavior():\n    assert signature(target).parameters['x'].default == 3\n"
    )
    assert main([str(path)]) == 0
    assert capsys.readouterr().out == ""


def test_definitions_preserve_unrelated_imports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "test_sample.py"
    path.write_text(
        "import inspect\ndef helper():\n    pass\n"
        "def test_default():\n    def local_helper():\n        pass\n"
        "    assert inspect.signature(target).parameters['x'].default == 3\n"
    )
    assert main([str(path)]) == 1
    assert f"{path}:7: TST003:" in capsys.readouterr().out


@pytest.mark.parametrize("github_actions", ["false", "true"])
def test_severity_configuration_changes_exit_status_and_annotations(
    github_actions: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", github_actions)
    path = tmp_path / "test_sample.py"
    path.write_text("def test_bad():\n    assert True\n")
    config = tmp_path / "pyproject.toml"
    for fail, warn, status, level in [
        ('["TST002"]', "[]", 1, "error"),
        ("[]", '["TST002"]', 0, "warning"),
        ("[]", "[]", 1, "error"),
    ]:
        config.write_text(f"[tool.check-tests]\nfail = {fail}\nwarn = {warn}\n")
        assert main([str(path)]) == status
        output = capsys.readouterr().out
        assert "TST002:" in output
        if github_actions == "true":
            assert output.startswith(f"::{level} file=")


@pytest.mark.parametrize("config", ["[project]\nname = 'example'", "[tool.ruff]\n"])
def test_missing_checker_config_keeps_rules_blocking(
    config: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "pyproject.toml").write_text(config)
    path = tmp_path / "test_sample.py"
    path.write_text("def test_bad():\n    assert Config().timeout == 30\n")
    assert main([str(path)]) == 1
    assert "TST001:" in capsys.readouterr().out


@pytest.mark.parametrize(
    "config",
    [
        "[broken",
        "tool = []",
        "[tool]\ncheck-tests = []",
        '[tool.check-tests]\nfail = ["TST999"]',
        '[tool.check-tests]\nwarn = ["TST999"]',
        '[tool.check-tests]\nfail = ["TST002"]\nwarn = ["TST002"]',
        '[tool.check-tests]\nwarn = "TST002"',
        "[tool.check-tests]\nfail = true",
        "[tool.check-tests]\nwarn = [2]",
        "[tool.check-tests]\nfail = [[]]",
        '[tool.check-tests]\nwarning = ["TST002"]',
    ],
)
def test_invalid_configuration_fails_even_without_findings(
    config: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "pyproject.toml").write_text(config)
    assert main([]) == 1
    assert "pyproject.toml:1: error: invalid configuration:" in capsys.readouterr().out


@pytest.mark.parametrize("source", [None, "def test_bad(:", "# test-quality: ignore TST002"])
def test_warning_configuration_cannot_downgrade_input_errors(
    source: str | None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.check-tests]\nwarn = ["TST001", "TST002", "TST003", "TST004", "TST005"]\n'
    )
    path = tmp_path / "test_sample.py"
    if source is not None:
        path.write_text(source)
    assert main([str(path)]) == 1
    assert ": error:" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("source", "config", "expected"),
    [
        ("def test_ok():\n    assert fetch() == 3\n", "", (0, "🟢 Completed", 0, 0)),
        ("def test_bad():\n    assert True\n", "", (1, "🔴 Failed", 1, 0)),
        (
            "def test_bad():\n    assert True\n    assert package.__version__ == '1'\n",
            '[tool.check-tests]\nwarn = ["TST005"]',
            (1, "🔴 Failed", 1, 1),
        ),
        (
            "def test_bad():\n    assert True\n",
            '[tool.check-tests]\nwarn = ["TST002"]',
            (0, "🟢 Completed", 0, 1),
        ),
        ("def test_ok():\n    pass\n", "[broken", (1, "🔴 Failed", 1, 0)),
    ],
)
def test_comment_report_matches_diagnostics(
    source: str,
    config: str,
    expected: tuple[int, str, int, int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status, title, failures, warnings = expected
    path = tmp_path / "test_sample.py"
    path.write_text(source)
    (tmp_path / "pyproject.toml").write_text(config)
    output = tmp_path / "report.md"
    monkeypatch.setenv("TEST_QUALITY_REPORT", str(output))
    assert main([str(path)]) == status
    report = output.read_text()
    assert report.startswith(f"## 🔎 Test quality\n\n{title}\n")
    assert f"<summary>Failures ({failures})</summary>" in report
    assert f"<summary>Warnings ({warnings})</summary>" in report
    if not failures and not warnings:
        assert "<pre>" not in report
    elif config == "[broken":
        assert "pyproject.toml:1: error: invalid configuration:" in report
    else:
        assert f"{path}:2: TST002:" in report


def test_comment_report_escapes_source_and_bounds_large_findings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "<script>@someone.py"
    path.write_text("def test_bad():\n" + "    assert True\n" * 500)
    output = tmp_path / "report.md"
    monkeypatch.setenv("TEST_QUALITY_REPORT", str(output))
    assert main([str(path)]) == 1
    report = output.read_text()
    assert "<summary>Failures (500)</summary>" in report
    assert "&lt;script&gt;&#64;someone.py:2: TST002:" in report
    assert "<script>" not in report
    assert "@someone" not in report
    assert "remaining findings are in the run logs" in report
    assert len(report) < 45000
