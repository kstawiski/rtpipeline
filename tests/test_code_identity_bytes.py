"""File-based code identities must bind every byte they name."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

from rtpipeline import custom_structures_rtstruct as custom
from rtpipeline import dvh, radiomics_robustness as rr, stage_completion as sc


def _digest(sources):
    return hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def test_rs_custom_generator_byte_identity_with_4ad1c53(monkeypatch):
    root = Path(custom.__file__).resolve().parents[1]
    sources = []
    for relative in sorted(custom.RS_CUSTOM_GENERATOR_CODE_SOURCES):
        original = subprocess.check_output(["git", "show", f"4ad1c53:rtpipeline/{relative}"], cwd=root)
        assert (root / "rtpipeline" / relative).read_bytes() == original, relative
        sources.append({"path": relative, "sha256": hashlib.sha256(original).hexdigest()})
    monkeypatch.setattr(custom, "_RS_CUSTOM_GENERATOR_RESOLVED", False)
    monkeypatch.setattr(custom, "_RS_CUSTOM_GENERATOR_SHA256", None)
    assert custom.rs_custom_generator_sha256() == _digest(sources)


def test_repository_code_identity_functions_do_not_filter_bytes(tmp_path, monkeypatch):
    root = Path(rr.__file__).resolve().parent
    # Scan every package module, including nested functions. Discover file-code
    # identities by their source-list references; reject unreviewed new ones.
    expected = {
        ("custom_structures_rtstruct.py", "rs_custom_generator_sha256"),
        ("dvh.py", "_current_dvh_measurement_code_sha256"),
        ("radiomics_robustness.py", "_current_robustness_code_identity"),
        ("stage_completion.py", "_code_identity"),
    }
    found, violations = set(), []
    filters = {"sub", "subn", "replace", "translate", "strip", "lstrip", "rstrip",
               "split", "splitlines", "rsplit", "decode", "dedent", "getsource", "getsourcefile"}
    for path in sorted(root.rglob("*.py")):
        source = path.read_text()
        relative = path.relative_to(root).as_posix()
        if "VALIDATION_ONLY" in source:
            violations.append(f"{relative}: source-exclusion marker")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            references = {n.id if isinstance(n, ast.Name) else n.attr
                          for n in ast.walk(node) if isinstance(n, (ast.Name, ast.Attribute))}
            if not any(ref == "code_sources" or ref.endswith("CODE_SOURCES") for ref in references):
                continue
            found.add((relative, node.name))
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    name = child.func.attr if isinstance(child.func, ast.Attribute) else getattr(child.func, "id", "")
                    if name in filters:
                        violations.append(f"{relative}:{child.lineno}: {name}")
                elif isinstance(child, ast.Subscript) and isinstance(child.slice, ast.Slice):
                    violations.append(f"{relative}:{child.lineno}: byte slicing")
    assert found == expected
    assert not violations, violations

    # Exercise the real hash functions on synthetic files containing exclusion
    # markers, arbitrary binary bytes and whitespace. This also covers imported
    # hashing helpers: a filter hidden there would fail the raw-byte comparison.
    definitions = list(sc._STAGE_DEFINITIONS.values())
    names = set(custom.RS_CUSTOM_GENERATOR_CODE_SOURCES) | set(dvh.DVH_MEASUREMENT_CODE_SOURCES) | set(rr.ROBUSTNESS_DISPOSITION_CODE_SOURCES)
    names.update(name for definition in definitions for name in definition.code_sources)
    package = tmp_path / "rtpipeline"
    for name in names:
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"prefix\r\n# VALIDATION_ONLY_BEGIN\n" + name.encode() + b"\x00\xff\n# VALIDATION_ONLY_END\n  suffix \n")
    def expected_sources(names):
        return [{"path": name, "sha256": hashlib.sha256((package / name).read_bytes()).hexdigest()}
                for name in sorted(set(names))]
    for module in (custom, dvh, rr, sc):
        monkeypatch.setattr(module, "__file__", str(package / Path(module.__file__).name))
    monkeypatch.setattr(custom, "_RS_CUSTOM_GENERATOR_RESOLVED", False)
    monkeypatch.setattr(custom, "_RS_CUSTOM_GENERATOR_SHA256", None)
    monkeypatch.setattr(dvh, "_DVH_MEASUREMENT_CODE_RESOLVED", False)
    monkeypatch.setattr(dvh, "_DVH_MEASUREMENT_CODE_SHA256", None)
    assert custom.rs_custom_generator_sha256() == _digest(expected_sources(custom.RS_CUSTOM_GENERATOR_CODE_SOURCES))
    assert dvh._current_dvh_measurement_code_sha256() == _digest(expected_sources(dvh.DVH_MEASUREMENT_CODE_SOURCES))
    identity = rr._current_robustness_code_identity()
    assert identity["sources"] == expected_sources(rr.ROBUSTNESS_DISPOSITION_CODE_SOURCES)
    assert identity["sources_sha256"] == _digest(identity["sources"])
    for definition in definitions:
        identity = sc._code_identity(definition)
        assert identity["sources"] == expected_sources(definition.code_sources)
        assert identity["sources_sha256"] == _digest(identity["sources"])
