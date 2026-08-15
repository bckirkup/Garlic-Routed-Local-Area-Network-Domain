#!/usr/bin/env python3
"""Fast, conservative checks for Sonar patterns without upstream lint rules."""

from __future__ import annotations

import argparse
import ast
import io
import re
import shlex
import tokenize
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    rule: str
    message: str


_PIP_INSTALL = re.compile(r"(?<![\w-])(?:python(?:\s+-m)?\s+)?pip\s+install\b")
_UV_SYNC = re.compile(r"(?<![\w-])uv\s+sync\b")
_UV_PIP_INSTALL = re.compile(r"(?<![\w-])uv\s+pip\s+install\b")
_UV_ADD = re.compile(r"(?<![\w-])uv\s+add\b")
_UV_TOOL_INSTALL = re.compile(r"(?<![\w-])uv\s+tool\s+install\b")
_UV_TOOL_RUN = re.compile(r"(?<![\w-])uv\s+tool\s+run\b")
_UVX = re.compile(r"(?<![\w-])uvx\b")
_UV_RUN = re.compile(r"(?<![\w-])uv\s+run\b")
_INSTALLER_COMMAND = re.compile(
    "|".join(
        pattern.pattern
        for pattern in (
            _PIP_INSTALL,
            _UV_SYNC,
            _UV_PIP_INSTALL,
            _UV_ADD,
            _UV_TOOL_INSTALL,
            _UV_TOOL_RUN,
            _UVX,
            _UV_RUN,
        )
    )
)
_SHA256 = re.compile(r"(?:sha256=|#sha256=)[0-9a-fA-F]{64}")
_COMMIT_SHA = re.compile(r"@[0-9a-fA-F]{40}(?:[#/]|$)")
_VERSIONED_PACKAGE = re.compile(r"(?:===|==|~=|!=|>=|<=|>|<)")
_WORKFLOW_SUFFIXES = {".yml", ".yaml"}


def _float_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, float)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        return _float_literal(node.operand)
    return False


def _binding_kind(node: ast.AST) -> str:
    return "float" if _float_literal(node) else "other"


def _record_binding(bindings: dict[str, list[str]], target: ast.AST, kind: str) -> None:
    if isinstance(target, ast.Name):
        bindings.setdefault(target.id, []).append(kind)


def _local_nodes(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
        return []
    nodes = [node]
    for child in ast.iter_child_nodes(node):
        nodes.extend(_local_nodes(child))
    return nodes


def _record_argument_bindings(assignments: dict[str, list[str]], arguments: ast.arguments) -> None:
    for argument in (
        *arguments.posonlyargs,
        *arguments.args,
        *arguments.kwonlyargs,
    ):
        assignments.setdefault(argument.arg, []).append("other")
    if arguments.vararg is not None:
        assignments.setdefault(arguments.vararg.arg, []).append("other")
    if arguments.kwarg is not None:
        assignments.setdefault(arguments.kwarg.arg, []).append("other")


def _record_local_binding(assignments: dict[str, list[str]], node: ast.AST) -> None:
    if isinstance(node, ast.Assign | ast.AnnAssign):
        _record_assignment_binding(assignments, node)
    elif isinstance(node, ast.AugAssign | ast.For | ast.AsyncFor):
        _record_loop_binding(assignments, node)
    elif isinstance(node, ast.With | ast.AsyncWith):
        _record_context_binding(assignments, node)
    elif isinstance(node, ast.ExceptHandler):
        _record_exception_binding(assignments, node)
    elif isinstance(node, ast.NamedExpr):
        _record_binding(assignments, node.target, _binding_kind(node.value))


def _record_assignment_binding(
    assignments: dict[str, list[str]], node: ast.Assign | ast.AnnAssign
) -> None:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            _record_binding(assignments, target, _binding_kind(node.value))
        return
    kind = _binding_kind(node.value) if node.value is not None else "other"
    _record_binding(assignments, node.target, kind)


def _record_loop_binding(
    assignments: dict[str, list[str]], node: ast.AugAssign | ast.For | ast.AsyncFor
) -> None:
    _record_binding(assignments, node.target, "other")


def _record_context_binding(
    assignments: dict[str, list[str]], node: ast.With | ast.AsyncWith
) -> None:
    for item in node.items:
        if item.optional_vars is not None:
            _record_binding(assignments, item.optional_vars, "other")


def _record_exception_binding(assignments: dict[str, list[str]], node: ast.ExceptHandler) -> None:
    if node.name is not None:
        assignments.setdefault(node.name, []).append("other")


def _local_float_bindings(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    assignments: dict[str, list[str]] = {}
    _record_argument_bindings(assignments, node.args)
    for statement in node.body:
        for child in _local_nodes(statement):
            _record_local_binding(assignments, child)
    return {name for name, kinds in assignments.items() if len(kinds) == 1 and kinds[0] == "float"}


def _approx_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "approx"
    )


def _assertion_float_finding(node: ast.Assert, path: Path, bindings: set[str]) -> Finding | None:
    for comparison in ast.walk(node.test):
        if not isinstance(comparison, ast.Compare):
            continue
        if not any(isinstance(op, ast.Eq | ast.NotEq) for op in comparison.ops):
            continue
        operands = [comparison.left, *comparison.comparators]
        if any(
            _float_literal(operand) or isinstance(operand, ast.Name) and operand.id in bindings
            for operand in operands
        ) and not any(_approx_call(operand) for operand in operands):
            return Finding(
                path,
                comparison.lineno,
                "S1244",
                "float equality in assert; use pytest.approx",
            )
    return None


def _walk_assertions(
    node: ast.AST, path: Path, bindings: set[str], findings: list[Finding]
) -> None:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        local_bindings = _local_float_bindings(node)
        for statement in node.body:
            _walk_assertions(statement, path, local_bindings, findings)
        return
    if isinstance(node, ast.ClassDef):
        for statement in node.body:
            _walk_assertions(statement, path, set(), findings)
        return
    if isinstance(node, ast.Assert):
        finding = _assertion_float_finding(node, path, bindings)
        if finding is not None:
            findings.append(finding)
        return
    for child in ast.iter_child_nodes(node):
        _walk_assertions(child, path, bindings, findings)


def _assert_float_comparisons(tree: ast.AST, path: Path) -> list[Finding]:
    findings: list[Finding] = []
    _walk_assertions(tree, path, set(), findings)
    return findings


def _safe_path(path: Path) -> Path:
    base = Path.cwd().resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(base):
        raise ValueError(f"path escapes the repository: {path}")
    return resolved


def _relative_path(path: Path) -> Path:
    return path.relative_to(Path.cwd().resolve())


def _composite_assertions(tree: ast.AST, path: Path) -> list[Finding]:
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert) and any(
            isinstance(child, ast.BoolOp) and isinstance(child.op, ast.And)
            for child in ast.walk(node.test)
        ):
            findings.append(
                Finding(
                    path,
                    node.lineno,
                    "S9073",
                    "composite assertion; assert each condition separately",
                )
            )
    return findings


def _numpy_random_imports(tree: ast.AST) -> set[str]:
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level != 0 or node.module != "numpy.random":
            continue
        imported_names.update(
            alias.asname or alias.name for alias in node.names if alias.name != "*"
        )
    return imported_names


def _random_call_name(call: ast.Call, imported_names: set[str]) -> str | None:
    function = call.func
    if isinstance(function, ast.Name) and function.id in imported_names:
        return function.id
    if not isinstance(function, ast.Attribute):
        return None
    random_module = function.value
    if not (
        isinstance(random_module, ast.Attribute)
        and random_module.attr == "random"
        and isinstance(random_module.value, ast.Name)
        and random_module.value.id == "np"
    ):
        return None
    return function.attr


def _bare_random_calls(tree: ast.AST, path: Path) -> list[Finding]:
    findings: list[Finding] = []
    imported_names = _numpy_random_imports(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _random_call_name(node, imported_names)
        if name is None:
            continue
        if name == "default_rng":
            has_seed = bool(node.args) or any(keyword.arg == "seed" for keyword in node.keywords)
            if has_seed:
                continue
        findings.append(
            Finding(
                path,
                node.lineno,
                "S6709",
                f"bare NumPy random call {name!r}; use a seeded Generator",
            )
        )
    return findings


def _comment_lines(source: str) -> set[int]:
    comments: set[int] = set()
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    with suppress(tokenize.TokenError):
        comments.update(token.start[0] for token in tokens if token.type == tokenize.COMMENT)
    return comments


def _uncommented_passes(tree: ast.AST, source: str, path: Path) -> list[Finding]:
    comments = _comment_lines(source)
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Pass):
            continue
        if node.lineno in comments or node.lineno - 1 in comments:
            continue
        findings.append(
            Finding(path, node.lineno, "S1186", "empty pass stub needs an explanatory comment")
        )
    return findings


def _is_test_file(path: Path) -> bool:
    return "tests" in path.parts or path.name.startswith("test_") or path.name.endswith("_test.py")


def _check_python_file(path: Path) -> list[Finding]:
    path = _safe_path(path)
    display_path = _relative_path(path)
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(display_path))
    except (OSError, SyntaxError, UnicodeDecodeError) as error:
        print(f"{display_path}: unable to inspect ({error})")
        return []
    findings = [
        *_assert_float_comparisons(tree, display_path),
        *_bare_random_calls(tree, display_path),
        *_uncommented_passes(tree, source, display_path),
    ]
    if _is_test_file(path):
        findings.extend(_composite_assertions(tree, display_path))
    return findings


def _python_files(paths: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            files.add(path)
        elif path.is_dir():
            files.update(candidate for candidate in path.rglob("*.py") if candidate.is_file())
    return sorted(files)


def _workflow_files(paths: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        if path.is_file() and path.suffix in _WORKFLOW_SUFFIXES:
            files.add(path)
        elif path.is_dir():
            files.update(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix in _WORKFLOW_SUFFIXES
            )
    return sorted(files)


def _workflow_commands(source: str) -> list[tuple[int, str]]:
    lines = source.splitlines()
    commands: list[tuple[int, str]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not _INSTALLER_COMMAND.search(line):
            index += 1
            continue
        start = index
        command_lines = [line]
        while command_lines[-1].rstrip().endswith("\\") and index + 1 < len(lines):
            index += 1
            command_lines.append(lines[index])
        commands.append((start + 1, "\n".join(command_lines)))
        index += 1
    return commands


def _installer_command_segments(command: str) -> list[str]:
    segments: list[str] = []
    for match in _INSTALLER_COMMAND.finditer(command):
        segment = command[match.start() :]
        separator = re.search(r"&&|;", segment)
        if separator is not None:
            segment = segment[: separator.start()]
        segments.append(segment.replace("\\\n", " "))
    return segments


def _local_target(target: str) -> bool:
    package = target.split("[", maxsplit=1)[0]
    if package.startswith((".", "/")):
        return True
    return Path(package).exists()


def _editable_install_is_local(tokens: list[str]) -> bool:
    targets: list[str] = []
    editable = False
    index = 0
    while index < len(tokens):
        arg = tokens[index]
        if arg in {"-e", "--editable"}:
            editable = True
            if index + 1 < len(tokens):
                targets.append(tokens[index + 1])
                index += 2
                continue
        index += 1
    return editable and bool(targets) and all(_local_target(target) for target in targets)


def _requirement_tokens(tokens: list[str]) -> list[str]:
    option_values = {
        "--config-settings",
        "--constraint",
        "-c",
        "--extra-index-url",
        "--find-links",
        "-f",
        "--index-url",
        "--only-binary",
        "--platform",
        "--python-version",
        "--requirement",
        "-r",
        "--target",
        "-t",
        "--trusted-host",
    }
    requirements: list[str] = []
    index = 0
    while index < len(tokens):
        arg = tokens[index]
        if arg in {"-e", "--editable"}:
            index += 2
            continue
        if arg in option_values:
            index += 2
            continue
        if arg.startswith("-"):
            index += 1
            continue
        requirements.append(arg)
        index += 1
    return requirements


def _requirement_is_pinned(requirement: str) -> bool:
    return bool(
        _SHA256.search(requirement)
        or _COMMIT_SHA.search(requirement)
        or _VERSIONED_PACKAGE.search(requirement)
    )


def _uv_command_kind(segment: str) -> str | None:
    for kind, pattern in (
        ("sync", _UV_SYNC),
        ("pip_install", _UV_PIP_INSTALL),
        ("add", _UV_ADD),
        ("tool_install", _UV_TOOL_INSTALL),
        ("tool_run", _UV_TOOL_RUN),
        ("tool_run", _UVX),
        ("run", _UV_RUN),
    ):
        if pattern.match(segment):
            return kind
    return None


def _uv_from_requirements(tokens: list[str]) -> list[str]:
    requirements: list[str] = []
    index = 0
    while index < len(tokens):
        arg = tokens[index]
        if arg == "--from" and index + 1 < len(tokens):
            requirements.append(tokens[index + 1])
            index += 2
            continue
        if arg.startswith("--from="):
            requirements.append(arg.removeprefix("--from="))
        index += 1
    return requirements


def _uv_build_finding(path: Path, line: int, tokens: list[str]) -> list[Finding]:
    if "--no-build" not in tokens:
        return [
            Finding(
                path, line, "S8541", "uv command must include --no-build for source-build safety"
            )
        ]
    return []


def _uv_sync_findings(path: Path, line: int, tokens: list[str]) -> list[Finding]:
    if "--locked" not in tokens and "--frozen" not in tokens:
        return [
            Finding(
                path,
                line,
                "S8544",
                "uv sync must include --locked or --frozen for pinned resolution",
            )
        ]
    return []


def _uv_tool_run_findings(path: Path, line: int, tokens: list[str]) -> list[Finding]:
    requirements = _uv_from_requirements(tokens)
    if not requirements or not all(_requirement_is_pinned(req) for req in requirements):
        return [
            Finding(
                path,
                line,
                "S8544",
                "uv tool commands must use a pinned --from requirement",
            )
        ]
    return []


def _uv_install_findings(path: Path, line: int, tokens: list[str]) -> list[Finding]:
    from_requirements = _uv_from_requirements(tokens)
    if from_requirements:
        requirements = from_requirements
    else:
        local_install = _editable_install_is_local(tokens)
        if local_install or "--require-hashes" in tokens:
            return []
        requirements = _requirement_tokens(tokens)
    if requirements and not all(_requirement_is_pinned(req) for req in requirements):
        return [
            Finding(
                path,
                line,
                "S8544",
                "uv install requirements must be version-pinned or hashed",
            )
        ]
    return []


def _uv_resolution_findings(path: Path, line: int, kind: str, tokens: list[str]) -> list[Finding]:
    if kind == "sync":
        return _uv_sync_findings(path, line, tokens)
    if kind == "tool_run":
        return _uv_tool_run_findings(path, line, tokens)
    if kind in {"pip_install", "add", "tool_install"}:
        return _uv_install_findings(path, line, tokens)
    return []


def _uv_findings(path: Path, line: int, kind: str, tokens: list[str]) -> list[Finding]:
    return _uv_build_finding(path, line, tokens) + _uv_resolution_findings(path, line, kind, tokens)


def _uv_command_tokens(kind: str, tokens: list[str]) -> list[str]:
    if kind == "tool_run" and tokens and tokens[0] == "uvx":
        return tokens[1:]
    command_index = next(
        (index for index, arg in enumerate(tokens) if arg in {"sync", "add", "install", "run"}),
        len(tokens),
    )
    return tokens[command_index + 1 :]


def _workflow_findings(path: Path) -> list[Finding]:
    base = Path.cwd().resolve()
    path = path.resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"path escapes the repository: {path}")
    display_path = _relative_path(path)
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        print(f"{display_path}: unable to inspect ({error})")
        return []
    return [
        finding
        for line, command in _workflow_commands(source)
        for segment in _installer_command_segments(command)
        for finding in _workflow_segment_findings(display_path, line, segment)
    ]


def _workflow_segment_findings(path: Path, line: int, segment: str) -> list[Finding]:
    try:
        tokens = shlex.split(segment)
    except ValueError:
        tokens = segment.split()
    if _PIP_INSTALL.match(segment):
        return _pip_findings(path, line, tokens)
    kind = _uv_command_kind(segment)
    if kind is None:
        return []
    return _uv_findings(path, line, kind, _uv_command_tokens(kind, tokens))


def _pip_findings(path: Path, line: int, tokens: list[str]) -> list[Finding]:
    install_index = next(
        (index for index, arg in enumerate(tokens) if arg == "install"),
        len(tokens),
    )
    install_tokens = tokens[install_index + 1 :]
    local_install = _editable_install_is_local(install_tokens)
    findings: list[Finding] = []
    if "--only-binary" not in install_tokens and not local_install:
        findings.append(
            Finding(
                path,
                line,
                "S8541",
                "pip install must include --only-binary :all: for published packages",
            )
        )
    if local_install or "--require-hashes" in install_tokens:
        return findings
    requirements = _requirement_tokens(install_tokens)
    if requirements and not all(_requirement_is_pinned(req) for req in requirements):
        findings.append(
            Finding(
                path,
                line,
                "S8544",
                "pip install requirements must be version-pinned or hashed",
            )
        )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflows",
        action="store_true",
        help="scan workflow YAML files instead of Python files",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="files or directories to scan",
    )
    args = parser.parse_args()
    try:
        requested_paths = [_safe_path(path) for path in args.paths]
    except ValueError as error:
        parser.error(str(error))
    if args.workflows:
        paths = requested_paths or [_safe_path(Path(".github/workflows"))]
        findings = [
            finding for path in _workflow_files(paths) for finding in _workflow_findings(path)
        ]
    else:
        paths = requested_paths or [
            _safe_path(Path("src")),
            _safe_path(Path("tests")),
            _safe_path(Path("scripts")),
            _safe_path(Path("baselines")),
        ]
        findings = [
            finding for path in _python_files(paths) for finding in _check_python_file(path)
        ]
        workflow_paths = [path for path in requested_paths if path.suffix in _WORKFLOW_SUFFIXES]
        findings.extend(
            finding
            for path in _workflow_files(workflow_paths)
            for finding in _workflow_findings(path)
        )
    for finding in findings:
        print(f"{finding.path}:{finding.line}:1: {finding.rule} {finding.message}")
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
