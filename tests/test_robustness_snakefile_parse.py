from __future__ import annotations

import ast
import io
import types
from pathlib import Path
from typing import Any

import pytest
from snakemake.parser import parse

SNAKEFILE_PATH = Path(__file__).resolve().parents[1] / "Snakefile"


class _InMemorySourcePath:
    """Read-only in-memory SourceFile adapter satisfying Snakemake parser interface."""

    def __init__(self, path_str: str) -> None:
        self._path_str = path_str

    def get_path_or_uri(self, secret_free: bool = True) -> str:
        return self._path_str


class _InMemorySourceCache:
    """In-memory sourcecache adapter returning io.StringIO of exact input bytes."""

    def __init__(self, content: str) -> None:
        self._content = content

    def open(self, path: Any) -> io.StringIO:
        return io.StringIO(self._content)


class _InMemoryWorkflow:
    """Minimal workflow adapter providing sourcecache for offline parsing."""

    def __init__(self, content: str) -> None:
        self.sourcecache = _InMemorySourceCache(content)


def parse_snakefile_source(
    content: str, path_str: str = str(SNAKEFILE_PATH)
) -> tuple[str, int, dict[int, int]]:
    """Parse Snakefile content with the installed Snakemake parser.

    Uses snakemake.parser.parse without initializing a full Workflow or executing
    any top-level statements.
    """
    linemap: dict[int, int] = {}
    path_adapter = _InMemorySourcePath(path_str)
    workflow_adapter = _InMemoryWorkflow(content)
    compiled_text, rulecount = parse(
        path=path_adapter,
        workflow=workflow_adapter,
        linemap=linemap,
        overwrite_shellcmd=None,
        rulecount=0,
    )
    return compiled_text, rulecount, linemap


def test_current_snakefile_parses_and_compiles_offline() -> None:
    """Verify current Snakefile grammar parses and compiles with compile() only.

    Crucially: NEVER exec() or eval() the compiled code, avoiding top-level side effects.
    """
    assert SNAKEFILE_PATH.is_file(), f"Snakefile not found at {SNAKEFILE_PATH}"
    content = SNAKEFILE_PATH.read_text(encoding="utf-8")
    assert len(content) > 0, "Snakefile is unexpectedly empty"

    compiled_py, rulecount, linemap = parse_snakefile_source(content)

    assert rulecount == 15, f"Expected 15 rules in Snakefile, got {rulecount}"
    assert len(compiled_py) > 0, "Parser returned empty compiled python string"
    assert len(linemap) > 0, "Parser returned empty linemap"

    # Compile with Python built-in compile() only. NEVER exec() or eval().
    code_obj = compile(compiled_py, str(SNAKEFILE_PATH), "exec")
    assert isinstance(code_obj, types.CodeType), "compile() must return a valid CodeType"


def test_ast_aggregate_radiomics_robustness_contract() -> None:
    """Verify AST structure of aggregate_radiomics_robustness rule.

    Asserts:
    1. Generated function carries expected parameters, including 'params'.
    2. Decorators include @workflow.params with expected keys.
    3. 'raw_values' parameter is present as an ast.Lambda with args (w, output).
    4. Other essential params (output_dir, root_dir, configfile, robustness_enabled, python, python_bin)
       are properly declared.
    """
    content = SNAKEFILE_PATH.read_text(encoding="utf-8")
    compiled_py, _, _ = parse_snakefile_source(content)
    tree = ast.parse(compiled_py, filename=str(SNAKEFILE_PATH))

    agg_func = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "__rule_aggregate_radiomics_robustness"
        ):
            agg_func = node
            break

    assert agg_func is not None, "Rule function __rule_aggregate_radiomics_robustness not found in AST"

    # Verify function arguments include 'params'
    func_args = [arg.arg for arg in agg_func.args.args]
    assert "params" in func_args, f"Expected 'params' in function args, got {func_args}"
    assert "input" in func_args
    assert "output" in func_args
    assert "wildcards" in func_args
    assert "threads" in func_args
    assert "log" in func_args

    # Inspect decorators on the rule function
    params_decorator = None
    rule_decorator = None
    for dec in agg_func.decorator_list:
        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
            if dec.func.attr == "params":
                params_decorator = dec
            elif dec.func.attr == "rule":
                rule_decorator = dec

    assert rule_decorator is not None, "@workflow.rule decorator missing on aggregate_radiomics_robustness"
    assert params_decorator is not None, "@workflow.params decorator missing on aggregate_radiomics_robustness"

    # Verify params keywords
    params_dict = {kw.arg: kw.value for kw in params_decorator.keywords}
    expected_keys = {
        "output_dir",
        "root_dir",
        "configfile",
        "robustness_enabled",
        "raw_values",
        "python",
        "python_bin",
    }
    for key in expected_keys:
        assert key in params_dict, f"Expected param '{key}' in aggregate_radiomics_robustness params, found {list(params_dict.keys())}"

    # Verify raw_values is a lambda with (w, output)
    raw_values_node = params_dict["raw_values"]
    assert isinstance(raw_values_node, ast.Lambda), f"Expected raw_values to be ast.Lambda, got {type(raw_values_node)}"
    lambda_args = [arg.arg for arg in raw_values_node.args.args]
    assert lambda_args == ["w", "output"], f"Expected lambda args ['w', 'output'], got {lambda_args}"


def test_ast_radiomics_robustness_course_rules() -> None:
    """Verify AST structure of both radiomics_robustness_course rules (container and local branches)."""
    content = SNAKEFILE_PATH.read_text(encoding="utf-8")
    compiled_py, _, _ = parse_snakefile_source(content)
    tree = ast.parse(compiled_py, filename=str(SNAKEFILE_PATH))

    course_funcs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "__rule_radiomics_robustness_course"
    ]

    assert len(course_funcs) == 2, (
        f"Expected exactly 2 definitions of __rule_radiomics_robustness_course (container and local), "
        f"found {len(course_funcs)}"
    )

    for idx, fn in enumerate(course_funcs):
        func_args = [arg.arg for arg in fn.args.args]
        assert "params" in func_args, f"Course rule {idx} missing 'params' in args: {func_args}"

        decorators = {}
        for dec in fn.decorator_list:
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                decorators[dec.func.attr] = dec

        assert "rule" in decorators, f"Course rule {idx} missing @workflow.rule"
        assert "conda" in decorators, f"Course rule {idx} missing @workflow.conda"
        assert "params" in decorators, f"Course rule {idx} missing @workflow.params"

        params_kw = {kw.arg: kw.value for kw in decorators["params"].keywords}
        assert "enabled" in params_kw, f"Course rule {idx} missing 'enabled' param"
        assert "config" in params_kw, f"Course rule {idx} missing 'config' param"
        assert "python" in params_kw, f"Course rule {idx} missing 'python' param"
        assert "course_dir" in params_kw, f"Course rule {idx} missing 'course_dir' param"
        assert "parquet" in params_kw, f"Course rule {idx} missing 'parquet' param"

        assert isinstance(params_kw["course_dir"], ast.Lambda), f"Course rule {idx} course_dir must be Lambda"
        assert isinstance(params_kw["parquet"], ast.Lambda), f"Course rule {idx} parquet must be Lambda"


def test_no_parsing_skipped_across_grammar() -> None:
    """Verify that all expected rules are parsed and none are silently skipped."""
    content = SNAKEFILE_PATH.read_text(encoding="utf-8")
    compiled_py, rulecount, _ = parse_snakefile_source(content)
    tree = ast.parse(compiled_py, filename=str(SNAKEFILE_PATH))

    parsed_rule_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "rule":
                for kw in node.keywords:
                    if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                        parsed_rule_names.append(kw.value.value)

    assert rulecount == len(parsed_rule_names) == 15

    expected_rules = {
        "all",
        "organize_courses",
        "segmentation_course",
        "crop_ct_course",
        "dvh_course",
        "qc_course",
        "aggregate_radiomics_robustness",
        "aggregate_results",
        "generate_report",
        "segmentation_custom_models",
        "radiomics_course",
        "radiomics_robustness_course",
    }
    assert set(parsed_rule_names) == expected_rules


@pytest.mark.parametrize(
    "corrupted_snippet,error_type,error_pattern",
    [
        (
            "rule invalid_directive:\n    nonexistent_directive: 'invalid'\n",
            SyntaxError,
            "Unexpected keyword",
        ),
        (
            "rule unclosed_bracket:\n    input:\n        ['unclosed'\n",
            SyntaxError,
            "EOF in multi-line statement",
        ),
        (
            "rule invalid_python:\n    params:\n        bad = 1 + * 2\n",
            SyntaxError,
            "invalid syntax",
        ),
    ],
)
def test_negative_controls_detect_corrupted_syntax(
    corrupted_snippet: str, error_type: type[Exception], error_pattern: str
) -> None:
    """Ensure parser/compiler detects invalid syntax and fails closed.

    This negative control proves a no-op parser or superficial mock cannot pass.
    """
    with pytest.raises(error_type) as exc_info:
        compiled_text, _, _ = parse_snakefile_source(corrupted_snippet, path_str="synthetic.smk")
        compile(compiled_text, "synthetic.smk", "exec")

    assert error_pattern in str(exc_info.value)
