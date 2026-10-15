"""Fast structural tests for the fail-closed release manifest."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / ".github" / "release-files.json"
VERIFIER_PATH = ROOT / "tools" / "verify_release.py"


def _verifier_module():
    spec = importlib.util.spec_from_file_location("release_verifier", VERIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _relative_files(directory: str):
    root = ROOT / directory
    # Recursive: the shipped manifest carries nested paths (for example
    # benchmarks/AB_VERIFIED_2026-04-14.md and
    # data/third_party/karl-keysingularity/README.md), so a top-level-only
    # walk cannot see a new public file below the first level and would
    # silently accept it not shipping.
    files = {
        path.relative_to(ROOT).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name != ".DS_Store"
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }
    if not (ROOT / ".git").exists():
        # An archive has no Git exclusions. Verify a clean source export here.
        return files

    # Respect Git exclusions when determining the source inventory.
    # Include both tracked files and new public files awaiting staging; an
    # ignored tracked file must still be checked. Run Git without a shell and
    # disable any configured filesystem-monitor command for this read.
    source = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "-C", str(ROOT),
         "ls-files", "--cached", "--others",
         "--exclude-standard", "-z", "--", directory],
        check=True,
        capture_output=True,
    )
    public_paths = {
        path.decode("utf-8") for path in source.stdout.split(b"\0") if path
    }
    return files & public_paths


def test_release_manifest_is_canonical_and_self_consistent():
    import connes_cvs

    module = _verifier_module()
    manifest = module._load_manifest(MANIFEST_PATH)
    assert manifest["distribution"] == "connes-cvs"
    assert manifest["version"] == connes_cvs.__version__, (
        f"release manifest version ({manifest['version']}) must match "
        f"connes_cvs.__version__ ({connes_cvs.__version__}); pyproject and "
        "installer metadata are bound to both in the verifier and the "
        "c13 regression suite"
    )
    assert ".github/release-files.json" in manifest["sdist_files"]
    assert "tools/verify_release.py" in manifest["sdist_files"]
    assert len(manifest["sdist_files"]) == len(set(manifest["sdist_files"]))
    assert len(manifest["wheel_files"]) == len(set(manifest["wheel_files"]))


def test_every_package_test_example_and_public_data_file_is_allowlisted():
    sdist = set(_manifest()["sdist_files"])
    for directory in (
        "connes_cvs",
        "benchmarks",
        "tools",
        "data",
        "data/c100",
        "data/third_party/karl-keysingularity",
    ):
        assert _relative_files(directory) <= sdist, directory
    assert not any(name.startswith("papers/") for name in sdist)

    expected_tests = _relative_files("tests")
    expected_examples = _relative_files("examples")
    assert expected_tests <= sdist
    assert expected_examples <= sdist

    required = {
        "data/c100/c100_N150_gamma_extraction_retight.json",
        "data/third_party/karl-keysingularity/2026-08-02_c13_validation_artifact.json",
        "examples/make_fig9_c100_aitken.py",
        "examples/make_fig10_c100_gamma_digits.py",
        "tests/test_validation.py",
    }
    assert required <= sdist
    assert not any(name.startswith("data/paper2/") for name in sdist)
    assert not any("paper2" in Path(name).name for name in sdist if name.startswith("examples/"))


def test_source_configuration_agrees_with_distribution_manifest():
    # The release verifier parses TOML with the standard library. Python 3.10
    # still runs the inventory tests; the 3.11+ legs also check both allowlists.
    pytest.importorskip("tomllib")
    verifier = _verifier_module()
    verifier._verify_source_configuration(ROOT, _manifest())


def test_wheel_manifest_is_exactly_the_public_package_plus_metadata():
    manifest = _manifest()
    wheel = set(manifest["wheel_files"])
    assert not any(name.startswith("papers/") for name in wheel)
    package_files = _relative_files("connes_cvs")
    assert {name for name in wheel if name.startswith("connes_cvs/")} == (
        package_files
    )
    assert len(wheel - package_files) == 4


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "a/../../escape",
        "a\\windows-path",
        "a/./noncanonical",
        "nul\x00byte",
        "control\nname",
    ],
)
def test_release_verifier_rejects_unsafe_archive_paths(name):
    module = _verifier_module()
    with pytest.raises(module.ReleaseVerificationError):
        module._validate_member_name(name)


def test_release_verifier_rejects_casefold_collisions():
    module = _verifier_module()
    with pytest.raises(module.ReleaseVerificationError, match="collision"):
        module._validate_name_collection(["Data/x", "data/x"], [])


def test_release_verifier_rejects_an_exact_manifest_extra():
    module = _verifier_module()
    with pytest.raises(module.ReleaseVerificationError, match="extra"):
        module._check_exact(["expected", "unexpected"], ["expected"], "test")


def test_dependency_floors_remain_pinned():
    """
    Guard the declared floors against silent drift.

    The Python floor is 3.10, not 3.9: the pinned build backend
    (``hatchling==1.32.0``, which keeps the release archives byte-reproducible)
    itself requires 3.10 or newer, so a source build on 3.9 cannot succeed.
    Python 3.9 reached end of life in October 2025; installations that need it
    should pin ``connes-cvs==0.2.2``.
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.10"' in text
    assert '"mpmath>=1.3.0"' in text
    assert text.count('"python-flint>=0.5.0"') == 3
    assert "python-flint>=0.8.0" not in text
