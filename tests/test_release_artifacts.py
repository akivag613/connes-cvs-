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


def _relative_files(directory: str, *, source_root: Path = ROOT):
    files = {
        path.relative_to(source_root).as_posix()
        for path in (source_root / directory).rglob("*")
        if path.is_file() and path.name != ".DS_Store"
    }
    if not (source_root / ".git").exists():
        # Source exports have no Git exclusions. Ignore only runtime bytecode
        # created by importing modules while testing the clean export.
        return {
            name for name in files
            if "__pycache__" not in Path(name).parts
            and Path(name).suffix not in {".pyc", ".pyo"}
        }

    # Include tracked files and new public files awaiting staging. Git's local
    # exclusions protect untracked working files, never tracked source files.
    source = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "-C", str(source_root),
         "ls-files", "--cached", "--others",
         "--exclude-standard", "-z", "--", directory],
        check=True,
        capture_output=True,
    )
    public_paths = {
        path.decode("utf-8") for path in source.stdout.split(b"\0") if path
    }
    return files & public_paths


def _assert_public_inventory(source_root: Path, sdist: set[str]):
    for directory in ("connes_cvs", "benchmarks", "tools", "data", "tests", "examples"):
        missing = _relative_files(directory, source_root=source_root) - sdist
        assert not missing, f"unlisted public files in {directory}: {sorted(missing)}"


def test_release_manifest_is_canonical_and_self_consistent():
    module = _verifier_module()
    manifest = module._load_manifest(MANIFEST_PATH)
    assert manifest["distribution"] == "connes-cvs"
    assert isinstance(manifest["version"], str) and manifest["version"]
    assert ".github/release-files.json" in manifest["sdist_files"]
    assert "tools/verify_release.py" in manifest["sdist_files"]
    assert len(manifest["sdist_files"]) == len(set(manifest["sdist_files"]))
    assert len(manifest["wheel_files"]) == len(set(manifest["wheel_files"]))


def test_every_package_test_example_and_public_data_file_is_allowlisted():
    sdist = set(_manifest()["sdist_files"])
    _assert_public_inventory(ROOT, sdist)
    assert not any(name.startswith("papers/") for name in sdist)
    required = {
        ".gitignore",
        "REFERENCES_TO_THIS_WORK.md",
        "data/c100/c100_N150_gamma_extraction_retight.json",
        "data/third_party/karl-keysingularity/2026-08-02_c13_validation_artifact.json",
        "examples/make_fig9_c100_aitken.py",
        "examples/make_fig10_c100_gamma_digits.py",
        "tests/test_figure_data_resolution.py",
        "tests/test_validation.py",
    }
    assert required <= sdist


def test_source_configuration_agrees_with_distribution_manifest():
    # The release verifier parses TOML with the standard library. Python 3.10
    # still runs the inventory tests; the 3.11+ legs also check both allowlists.
    pytest.importorskip("tomllib")
    verifier = _verifier_module()
    verifier._verify_source_configuration(ROOT, _manifest())


@pytest.fixture
def source_configuration(tmp_path):
    """Minimal source tree for configuration mutations, without package imports."""
    manifest = _manifest()
    for name in manifest["sdist_files"]:
        if name == "PKG-INFO":
            continue
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    for name in ("pyproject.toml", "connes_cvs/__init__.py"):
        (tmp_path / name).write_bytes((ROOT / name).read_bytes())
    return tmp_path, manifest


@pytest.mark.parametrize("target", ["project", "manifest", "package", "wheel_paths"])
def test_source_gate_rejects_a_partial_version_bump(source_configuration, target):
    pytest.importorskip("tomllib")
    root, manifest = source_configuration
    old = manifest["version"]
    new = old + ".post1"
    if target == "project":
        path = root / "pyproject.toml"
        path.write_text(path.read_text().replace(f'version = "{old}"', f'version = "{new}"'))
    elif target == "package":
        path = root / "connes_cvs" / "__init__.py"
        path.write_text(path.read_text().replace(f'__version__ = "{old}"', f'__version__ = "{new}"'))
    elif target == "manifest":
        manifest["version"] = new
    else:
        manifest["wheel_files"] = [
            name.replace(f"-{old}.dist-info/", f"-{new}.dist-info/")
            for name in manifest["wheel_files"]
        ]
    verifier = _verifier_module()
    with pytest.raises(verifier.ReleaseVerificationError, match="version|metadata paths"):
        verifier._verify_source_configuration(root, manifest)


def test_source_gate_accepts_a_synchronized_version_bump(source_configuration):
    pytest.importorskip("tomllib")
    root, manifest = source_configuration
    old = manifest["version"]
    new = old + ".post1"
    for name in ("pyproject.toml", "connes_cvs/__init__.py"):
        path = root / name
        path.write_text(path.read_text().replace(f'"{old}"', f'"{new}"'))
    manifest["version"] = new
    manifest["wheel_files"] = [
        name.replace(f"-{old}.dist-info/", f"-{new}.dist-info/")
        for name in manifest["wheel_files"]
    ]
    _verifier_module()._verify_source_configuration(root, manifest)


def test_source_gate_requires_gitignore_in_the_explicit_allowlist(source_configuration):
    pytest.importorskip("tomllib")
    root, manifest = source_configuration
    path = root / "pyproject.toml"
    path.write_text(path.read_text().replace('    "/.gitignore",\n', ""))
    verifier = _verifier_module()
    with pytest.raises(verifier.ReleaseVerificationError, match="sdist allowlist"):
        verifier._verify_source_configuration(root, manifest)


@pytest.mark.parametrize("name", [
    "data/nested/new.json",
    "examples/nested/paper2_demo.py",
    "tests/test_paper2_example.py",
    "connes_cvs/nested/helper.py",
    "benchmarks/nested/measurement.json",
    "tools/nested/helper.py",
])
def test_source_inventory_rejects_unlisted_files_at_any_depth(tmp_path, name):
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_text("example")
    with pytest.raises(AssertionError, match="unlisted public files"):
        _assert_public_inventory(tmp_path, set())
    _assert_public_inventory(tmp_path, {name})


def test_source_export_inventory_ignores_generated_bytecode(tmp_path):
    names = ["tests/__pycache__/sample.cpython-312.pyc", "tests/sample.pyc", "tests/.DS_Store"]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"generated")
    _assert_public_inventory(tmp_path, set())


def test_git_inventory_covers_tracked_and_untracked_public_files(tmp_path):
    def git(*args):
        return subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "-C", str(tmp_path), *args],
            check=True, capture_output=True,
        )

    git("init", "--quiet")
    tracked = "data/nested/tracked.json"
    public = "data/nested/public.json"
    ignored = "data/nested/ignored.json"
    for name in (tracked, public, ignored):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    git("add", "--", tracked)
    (tmp_path / ".git" / "info" / "exclude").write_text(f"/{tracked}\n/{ignored}\n")
    assert _relative_files("data", source_root=tmp_path) == {tracked, public}
    for allowed in (set(), {tracked}, {public}):
        with pytest.raises(AssertionError, match="unlisted public files"):
            _assert_public_inventory(tmp_path, allowed)
    _assert_public_inventory(tmp_path, {tracked, public})


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
