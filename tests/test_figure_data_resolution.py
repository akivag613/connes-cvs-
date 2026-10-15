"""Figure examples remain usable with absent, incomplete and relocated data."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "make_fig9_c100_aitken.py": (
        "results_c100_N100_T800_dps500_v020.json",
        "results_c100_N150_T800_dps500_v020.json",
        "results_c100_N200_T800_dps500_v020.json",
        "results_c100_N250_T800_dps500_v020.json",
    ),
    "make_fig10_c100_gamma_digits.py": (
        "c100_N150_gamma_extraction_retight.json",
        "c100_N150_dps1000_gamma_extraction.json",
        "c100_N250_dps500_gamma_extraction.json",
    ),
}


@pytest.fixture(params=EXAMPLES)
def example(request, tmp_path):
    """Import a stand-alone example tree containing no data directory."""
    script = tmp_path / "examples" / request.param
    script.parent.mkdir()
    shutil.copyfile(ROOT / "examples" / request.param, script)
    spec = importlib.util.spec_from_file_location(script.stem, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, script, EXAMPLES[request.param]


def _bundle(script: Path, files: tuple[str, ...], layout: str = "c100") -> Path:
    directory = script.parent.parent / "data" / layout
    directory.mkdir(parents=True)
    for filename in files:
        shutil.copyfile(ROOT / "data" / "c100" / filename, directory / filename)
    return directory


def _run(script: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(script.parent.parent.parent / "matplotlib-cache"))
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=script.parent.parent,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )


def test_import_and_help_do_not_require_data(example):
    _, script, _ = example
    result = _run(script, "--help")
    assert result.returncode == 0, result.stderr
    assert "--out" in result.stdout
    assert "Traceback" not in result.stderr
    assert not (script.parent.parent / "data").exists()


def test_missing_bundle_exits_cleanly(example):
    _, script, _ = example
    output = script.parent.parent / "figure.pdf"
    result = _run(script, "--out", str(output))
    assert result.returncode == 2
    assert "No complete c=100" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    "filename,missing",
    [(filename, missing) for filename, files in EXAMPLES.items() for missing in files],
)
def test_each_required_input_is_checked(filename, missing, tmp_path):
    script = tmp_path / "examples" / filename
    script.parent.mkdir()
    shutil.copyfile(ROOT / "examples" / filename, script)
    data = _bundle(script, EXAMPLES[filename])
    (data / missing).unlink()
    output = tmp_path / "figure.pdf"
    result = _run(script, "--out", str(output))
    assert result.returncode == 2
    assert f"{data}: missing {missing}" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_directory_is_not_accepted_in_place_of_input(example):
    module, script, files = example
    data = _bundle(script, files)
    (data / files[0]).unlink()
    (data / files[0]).mkdir()
    with pytest.raises(FileNotFoundError, match=files[0]):
        module._resolve_data_dir()


def test_complete_source_bundle_has_precedence(example):
    module, script, files = example
    source = _bundle(script, files)
    _bundle(script, files, "c100_verification")
    assert module._resolve_data_dir() == str(source)


def test_complete_zenodo_bundle_follows_incomplete_source(example):
    module, script, files = example
    source = _bundle(script, files)
    zenodo = _bundle(script, files, "c100_verification")
    for filename in files:
        (source / filename).unlink()
        assert module._resolve_data_dir() == str(zenodo)
        shutil.copyfile(zenodo / filename, source / filename)


def test_partial_bundles_are_not_combined(example):
    module, script, files = example
    source = _bundle(script, files)
    zenodo = _bundle(script, files, "c100_verification")
    (source / files[0]).unlink()
    for filename in files[1:]:
        (zenodo / filename).unlink()
    with pytest.raises(FileNotFoundError, match="No complete c=100"):
        module._resolve_data_dir()


@pytest.mark.parametrize("layout", ["c100", "c100_verification"])
def test_complete_bundle_generates_pdf(example, layout):
    _, script, files = example
    _bundle(script, files, layout)
    output = script.parent.parent / "output" / "figure.pdf"
    result = _run(script, "--out", str(output))
    assert result.returncode == 0, result.stderr
    assert "wrote " in result.stdout
    assert output.read_bytes().startswith(b"%PDF-")
