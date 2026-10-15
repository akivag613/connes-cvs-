#!/usr/bin/env python3
"""Fail-closed verifier for connes-cvs release archives.

Build first, then run::

    python -m build --sdist --wheel --outdir release-dist
    python tools/verify_release.py --dist release-dist

The checked-in manifest is authoritative: an added, omitted, renamed, unsafe,
or duplicate archive member fails verification.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import stat
import sys
import tarfile
import unicodedata
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Mapping, Sequence, Tuple


class ReleaseVerificationError(RuntimeError):
    """A release archive violated the checked-in contract."""


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseVerificationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseVerificationError(f"invalid release manifest: {exc}") from exc
    if manifest.get("schema") != "connes-cvs.release-files.v1":
        raise ReleaseVerificationError("unsupported release manifest schema")
    for field in ("sdist_files", "wheel_files"):
        values = manifest.get(field)
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ):
            raise ReleaseVerificationError(f"manifest {field} must be a string list")
        if values != sorted(values) or len(values) != len(set(values)):
            raise ReleaseVerificationError(
                f"manifest {field} must be sorted and duplicate-free"
            )
        for value in values:
            _validate_member_name(value)
    return manifest


def _validate_member_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ReleaseVerificationError("archive member name must be nonempty text")
    if "\\" in name or "\x00" in name or unicodedata.normalize("NFC", name) != name:
        raise ReleaseVerificationError(f"unsafe archive member name {name!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise ReleaseVerificationError(f"control character in archive path {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ReleaseVerificationError(f"unsafe archive member path {name!r}")
    if str(path) != name:
        raise ReleaseVerificationError(f"noncanonical archive member path {name!r}")


def _validate_name_collection(names: Sequence[str], forbidden: Iterable[str]) -> None:
    seen = set()
    portable_seen: Dict[str, str] = {}
    for name in names:
        _validate_member_name(name)
        if name in seen:
            raise ReleaseVerificationError(f"duplicate archive member {name!r}")
        seen.add(name)
        portable = unicodedata.normalize("NFC", name).casefold()
        prior = portable_seen.get(portable)
        if prior is not None:
            raise ReleaseVerificationError(
                f"portable path collision between {prior!r} and {name!r}"
            )
        portable_seen[portable] = name
        lowered = name.casefold()
        for fragment in forbidden:
            if fragment.casefold() in lowered:
                raise ReleaseVerificationError(
                    f"forbidden archive path fragment {fragment!r} in {name!r}"
                )


def _check_exact(actual: Sequence[str], expected: Sequence[str], label: str) -> None:
    actual_set = set(actual)
    expected_set = set(expected)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise ReleaseVerificationError(
            f"{label} manifest mismatch; missing={missing!r}, extra={extra!r}"
        )


def _check_limits(sizes: Mapping[str, int], manifest: Mapping[str, object]) -> None:
    limits = manifest["limits"]
    maximum_file = int(limits["maximum_file_bytes"])
    maximum_total = int(limits["maximum_total_bytes"])
    for name, size in sizes.items():
        if size < 0 or size > maximum_file:
            raise ReleaseVerificationError(
                f"archive member {name!r} has forbidden size {size}"
            )
    total = sum(sizes.values())
    if total > maximum_total:
        raise ReleaseVerificationError(f"archive expands to {total} bytes")


def _check_content(name: str, raw: bytes, forbidden: Iterable[str]) -> None:
    lowered = raw.lower()
    for fragment in forbidden:
        if fragment.encode("utf-8").lower() in lowered:
            raise ReleaseVerificationError(
                f"forbidden content fragment {fragment!r} in {name!r}"
            )


def _read_sdist(path: Path, manifest: Mapping[str, object]) -> Dict[str, bytes]:
    normalized = str(manifest["distribution"]).replace("-", "_")
    expected_prefix = f"{normalized}-{manifest['version']}/"
    payload: Dict[str, bytes] = {}
    sizes: Dict[str, int] = {}
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            members = archive.getmembers()
            raw_names = [member.name for member in members]
            _validate_name_collection(
                raw_names, manifest["forbidden_archive_fragments"]
            )
            for member in members:
                if not member.name.startswith(expected_prefix):
                    raise ReleaseVerificationError(
                        f"sdist member has wrong root prefix: {member.name!r}"
                    )
                relative = member.name[len(expected_prefix) :]
                _validate_member_name(relative)
                if not member.isfile() or member.issym() or member.islnk():
                    raise ReleaseVerificationError(
                        f"sdist member must be a regular file: {member.name!r}"
                    )
                if member.mode & 0o7000:
                    raise ReleaseVerificationError(
                        f"sdist member has privileged mode bits: {member.name!r}"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ReleaseVerificationError(
                        f"unable to read sdist member {member.name!r}"
                    )
                raw = extracted.read()
                if len(raw) != member.size:
                    raise ReleaseVerificationError(
                        f"sdist size mismatch for {member.name!r}"
                    )
                payload[relative] = raw
                sizes[relative] = member.size
    except (OSError, tarfile.TarError) as exc:
        raise ReleaseVerificationError(f"invalid sdist {path}: {exc}") from exc
    _check_limits(sizes, manifest)
    _check_exact(list(payload), manifest["sdist_files"], "sdist")
    for name, raw in payload.items():
        if name != ".github/release-files.json":
            _check_content(name, raw, manifest["forbidden_content_fragments"])
    return payload


def _read_wheel(path: Path, manifest: Mapping[str, object]) -> Dict[str, bytes]:
    payload: Dict[str, bytes] = {}
    sizes: Dict[str, int] = {}
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            _validate_name_collection(
                names, manifest["forbidden_archive_fragments"]
            )
            for info in infos:
                mode = info.external_attr >> 16
                if info.is_dir() or stat.S_ISLNK(mode):
                    raise ReleaseVerificationError(
                        f"wheel member must be a regular file: {info.filename!r}"
                    )
                raw = archive.read(info)
                if len(raw) != info.file_size:
                    raise ReleaseVerificationError(
                        f"wheel size mismatch for {info.filename!r}"
                    )
                payload[info.filename] = raw
                sizes[info.filename] = info.file_size
    except (OSError, zipfile.BadZipFile) as exc:
        raise ReleaseVerificationError(f"invalid wheel {path}: {exc}") from exc
    _check_limits(sizes, manifest)
    _check_exact(list(payload), manifest["wheel_files"], "wheel")
    for name, raw in payload.items():
        _check_content(name, raw, manifest["forbidden_content_fragments"])
    return payload


def _verify_metadata(raw: bytes, manifest: Mapping[str, object], label: str) -> None:
    metadata = BytesParser(policy=policy.default).parsebytes(raw)
    if metadata["Name"] != manifest["distribution"]:
        raise ReleaseVerificationError(f"{label} has wrong project name")
    if metadata["Version"] != manifest["version"]:
        raise ReleaseVerificationError(f"{label} has wrong version")
    if metadata["Requires-Python"] != ">=3.10":
        raise ReleaseVerificationError(f"{label} has wrong Python floor")
    requirements = metadata.get_all("Requires-Dist", [])
    if "mpmath>=1.3.0" not in requirements:
        raise ReleaseVerificationError(f"{label} lost the mpmath>=1.3.0 floor")


def _verify_wheel_record(payload: Mapping[str, bytes], record_name: str) -> None:
    try:
        rows = list(csv.reader(io.StringIO(payload[record_name].decode("utf-8"))))
    except (UnicodeError, csv.Error) as exc:
        raise ReleaseVerificationError(f"invalid wheel RECORD: {exc}") from exc
    if any(len(row) != 3 for row in rows):
        raise ReleaseVerificationError("wheel RECORD rows must have three fields")
    if len(rows) != len(payload):
        raise ReleaseVerificationError("wheel RECORD has wrong row count")
    recorded = {row[0]: row[1:] for row in rows}
    if len(recorded) != len(rows) or set(recorded) != set(payload):
        raise ReleaseVerificationError("wheel RECORD paths do not match archive")
    for name, raw in payload.items():
        hash_field, size_field = recorded[name]
        if name == record_name:
            if hash_field or size_field:
                raise ReleaseVerificationError("wheel RECORD must not hash itself")
            continue
        digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=")
        if hash_field != "sha256=" + digest.decode("ascii"):
            raise ReleaseVerificationError(f"wheel RECORD hash mismatch for {name}")
        if size_field != str(len(raw)):
            raise ReleaseVerificationError(f"wheel RECORD size mismatch for {name}")


def _verify_source_configuration(root: Path, manifest: Mapping[str, object]) -> None:
    try:
        import tomllib
    except ImportError as exc:  # pragma: no cover - release job uses Python 3.12
        raise ReleaseVerificationError(
            "release verification requires Python 3.11 or newer"
        ) from exc
    with (root / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    targets = project["tool"]["hatch"]["build"]["targets"]
    configured_sdist = targets["sdist"]["include"]
    # Only PKG-INFO is build-generated. .gitignore is a tracked source file and
    # must be allowlisted in pyproject like every other shipped path; hatchling
    # force-includes root VCS exclusion files regardless of the allowlist, and
    # dedups the member when it is listed, so naming it keeps both manifests in
    # exact agreement without changing the archive contents.
    generated = {"PKG-INFO"}
    expected_sdist = [
        "/" + name for name in manifest["sdist_files"] if name not in generated
    ]
    if sorted(configured_sdist) != expected_sdist or len(configured_sdist) != len(
        set(configured_sdist)
    ):
        raise ReleaseVerificationError(
            "pyproject sdist allowlist differs from the release manifest"
        )
    for entry in configured_sdist:
        if not (root / entry.lstrip("/")).is_file():
            raise ReleaseVerificationError(f"sdist include is not an exact file: {entry}")
    configured_wheel = targets["wheel"]["include"]
    expected_wheel_sources = [
        "/" + name
        for name in manifest["wheel_files"]
        if name.startswith("connes_cvs/")
    ]
    if configured_wheel != expected_wheel_sources:
        raise ReleaseVerificationError(
            "pyproject wheel allowlist differs from the release manifest"
        )
    if project["project"]["version"] != manifest["version"]:
        raise ReleaseVerificationError(
            "pyproject project version differs from the release manifest"
        )
    normalized = str(manifest["distribution"]).replace("-", "_")
    dist_info = f"{normalized}-{manifest['version']}.dist-info/"
    for name in manifest["wheel_files"]:
        if name.startswith(f"{normalized}/"):
            continue
        if not name.startswith(dist_info):
            raise ReleaseVerificationError(
                "wheel manifest metadata path does not carry the release version"
            )


def verify_release(root: Path, dist: Path) -> Tuple[Path, Path, int, int]:
    manifest = _load_manifest(root / ".github" / "release-files.json")
    _verify_source_configuration(root, manifest)
    normalized = manifest["distribution"].replace("-", "_")
    version = manifest["version"]
    sdist = dist / f"{normalized}-{version}.tar.gz"
    wheel = dist / f"{normalized}-{version}-py3-none-any.whl"
    if not dist.is_dir():
        raise ReleaseVerificationError(f"dist directory does not exist: {dist}")
    actual_artifacts = sorted(path.name for path in dist.iterdir())
    expected_artifacts = sorted([sdist.name, wheel.name])
    if actual_artifacts != expected_artifacts:
        raise ReleaseVerificationError(
            f"release directory must contain exactly {expected_artifacts!r}; "
            f"found {actual_artifacts!r}"
        )

    sdist_payload = _read_sdist(sdist, manifest)
    wheel_payload = _read_wheel(wheel, manifest)
    dist_info = f"{normalized}-{version}.dist-info"
    _verify_metadata(sdist_payload["PKG-INFO"], manifest, "sdist PKG-INFO")
    _verify_metadata(
        wheel_payload[f"{dist_info}/METADATA"], manifest, "wheel METADATA"
    )
    wheel_metadata = BytesParser(policy=policy.default).parsebytes(
        wheel_payload[f"{dist_info}/WHEEL"]
    )
    if wheel_metadata["Root-Is-Purelib"] != "true":
        raise ReleaseVerificationError("wheel is not marked purelib")
    if wheel_metadata.get_all("Tag", []) != ["py3-none-any"]:
        raise ReleaseVerificationError("wheel tag is not exactly py3-none-any")
    _verify_wheel_record(wheel_payload, f"{dist_info}/RECORD")

    for name in manifest["sdist_files"]:
        if name == "PKG-INFO":
            continue
        source = root / name
        if not source.is_file() or source.read_bytes() != sdist_payload[name]:
            raise ReleaseVerificationError(
                f"sdist payload differs from the source tree for {name}"
            )
    for name in manifest["wheel_files"]:
        if name.startswith("connes_cvs/") and wheel_payload[name] != sdist_payload[name]:
            raise ReleaseVerificationError(
                f"wheel and sdist package payload differ for {name}"
            )
    if wheel_payload[f"{dist_info}/licenses/LICENSE"] != sdist_payload["LICENSE"]:
        raise ReleaseVerificationError("wheel and sdist license payloads differ")

    return sdist, wheel, len(sdist_payload), len(wheel_payload)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    dist = args.dist.resolve()
    try:
        sdist, wheel, sdist_count, wheel_count = verify_release(root, dist)
    except ReleaseVerificationError as exc:
        print(f"release verification FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"sdist: {sdist.name} ({sdist_count} files, sha256={_sha256(sdist)})")
    print(f"wheel: {wheel.name} ({wheel_count} files, sha256={_sha256(wheel)})")
    print("release archive manifests, metadata, hashes, paths, and contents: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
