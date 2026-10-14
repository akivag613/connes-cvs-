"""
Regression gate for the c = 13 reference cells.

Fast tests (always run) cover the public API surface and the v0.3.0
``extract_zeros`` correctness fixes (the ``c=`` parameter and the
float64-``L`` UserWarning).

Slow tests (``pytest -m slow``) compute real c = 13 cells end-to-end and
enforce the repository regression contract on ``lambda_even`` against the
committed references in ``tests/reference_values.json``.  The per-leg
thresholds differ, because the references differ in strength:

- the A/B benchmark cell (c=13, N=80, T=400, dps=80), whose committed
  reference carries all 80 printed digits and was computed at dps = 80
  itself, so a same-precision recompute is deterministic.  This leg
  requires the FULL stored match: every one of the 80 reference digits,
  plus agreement of sign and decimal exponent.  There is no partial-digit
  floor on this cell.
- the quick cell documented in ``examples/basic_compute.py``
  (c=13, N=100, T=400, dps=80), checked against the stored 25-digit
  lambda reference (a dps = 150 value; dps = 80 arithmetic supports
  ~22 correct digits, so this leg gates at >= 20 to leave platform
  margin) and the 20-digit gamma_1-error reference (gated at >= 18).

The A/B leg compares decimal strings rather than raw ``_mpf_`` limbs on
purpose: the digamma backend (python-flint) may round a final unit in the
last place differently across backend versions, and the printed-digit
comparison is the widest contract that still pins every stored digit.

References
----------
- Connes & van Suijlekom, arXiv:2511.23257
- Connes, Consani & Moscovici, arXiv:2511.22755, Section 6
- data/results_15pt_T800.json (published 15-cutoff dataset)
"""

from __future__ import annotations

import json
import hashlib
import warnings
from pathlib import Path

import mpmath as mp
import pytest

REFERENCE_PATH = Path(__file__).resolve().parent / "reference_values.json"
KARL_ARTIFACT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "third_party"
    / "karl-keysingularity"
    / "2026-08-02_c13_validation_artifact.json"
)


def _load_reference(cell_key: str) -> dict:
    with open(REFERENCE_PATH) as fh:
        return json.load(fh)["cells"][cell_key]


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _decompose(s: str) -> tuple[int, int, str]:
    """Split a decimal string into ``(sign, exponent, mantissa digits)``.

    The value is ``sign * 0.<mantissa> * 10**(exponent + 1)``, i.e. the
    exponent is that of the leading significant digit in ``d.ddd e E``
    normal form.  Leading zeros are stripped; the returned mantissa is
    empty for an exact zero.
    """
    s = s.strip()
    sign = -1 if s.startswith("-") else 1
    s = s.lstrip("+-")
    exp = 0
    for marker in ("e", "E"):
        if marker in s:
            s, tail = s.split(marker, 1)
            exp = int(tail)
            break
    int_part, _, frac_part = s.partition(".")
    digits = int_part + frac_part
    stripped = digits.lstrip("0")
    if not stripped:
        return (0, 0, "")
    lead_zeros = len(digits) - len(stripped)
    # Position of the leading significant digit relative to the point.
    exponent = len(int_part) - lead_zeros - 1 + exp
    return (sign, exponent, stripped)


def _significant_digits(s: str) -> int:
    """Number of significant digits carried by a decimal string."""
    return len(_decompose(s)[2])


def _matching_digits(ref: str, got: str) -> int:
    """Count leading matching significant digits of two decimal strings.

    Sign and decimal exponent are compared first and a mismatch in either
    returns 0, so a sign-flipped or magnitude-shifted value can never be
    reported as a match on the strength of a shared mantissa prefix.
    Comparison stops at the shorter supplied mantissa. Missing digits are
    never invented by zero-padding.
    """
    for s in (ref, got):
        assert s, "empty value passed to _matching_digits"
    sa, ea, a = _decompose(ref)
    sb, eb, b = _decompose(got)
    if not a or not b:
        return 0
    if sa != sb or ea != eb:
        return 0
    n = min(len(a), len(b))
    m = 0
    for i in range(n):
        if a[i] == b[i]:
            m += 1
        else:
            break
    return m


def _has_flint() -> bool:
    try:
        import flint  # noqa: F401
        return True
    except ImportError:
        return False


# ============================================================
# Fast smoke tests
# ============================================================

def test_build_galerkin_matrix_callable():
    """Smoke test: build_galerkin_matrix is importable and callable."""
    from connes_cvs import build_galerkin_matrix
    assert callable(build_galerkin_matrix)


def test_compute_ground_state_callable():
    """Smoke test: compute_ground_state is importable and callable."""
    from connes_cvs import compute_ground_state
    assert callable(compute_ground_state)


def _is_within(child: Path, parent: Path) -> bool:
    """True when ``child`` is ``parent`` or lives underneath it."""
    return child == parent or parent in child.parents


def _editable_target(dist) -> Path | None:
    """Project root an editable installation points at, else ``None``.

    A ``pip install -e .`` records ``direct_url.json`` with
    ``dir_info.editable = true`` and a ``file://`` URL naming the checkout.
    For such an installation ``dist.locate_file`` still reports the (empty)
    site-packages location, so path identity alone can never confirm the
    binding; this recovers the real one.
    """
    try:
        raw = dist.read_text("direct_url.json")
    except (OSError, ValueError):
        raw = None
    if not raw:
        return None
    try:
        info = json.loads(raw)
    except ValueError:
        return None
    if not info.get("dir_info", {}).get("editable"):
        return None
    url = info.get("url") or ""
    if not url.startswith("file://"):
        return None
    from urllib.parse import unquote, urlparse

    return Path(unquote(urlparse(url).path)).resolve()


def test_package_imports():
    """The in-tree __version__ matches the installed distribution.

    Covers both installation shapes:

    - a regular (non-editable) install, where the imported ``__init__.py``
      is the installed one; and
    - an editable install (what CI and ``pip install -e '.[dev]'`` use),
      where the import resolves inside the checkout that
      ``direct_url.json`` points at.  Path identity never holds there, so
      an identity-only check silently skipped in CI.

    Only a pure source checkout with nothing installed, or an installed
    distribution unrelated to the imported package, skips.
    """
    import connes_cvs
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        dist = distribution("connes-cvs")
    except PackageNotFoundError:
        pytest.skip("connes-cvs is not installed (running from a source tree)")
    installed_init = Path(str(dist.locate_file("connes_cvs/__init__.py"))).resolve()
    imported_init = Path(connes_cvs.__file__).resolve()

    if installed_init != imported_init:
        target = _editable_target(dist)
        if target is None or not _is_within(imported_init, target):
            pytest.skip(
                "imported package is a source checkout that the installed "
                f"distribution ({dist.version}) does not point at; "
                "version-identity check not applicable"
            )

    assert connes_cvs.__version__ == dist.version, (
        f"connes_cvs.__version__ ({connes_cvs.__version__}) must match "
        f"installer-reported version ({dist.version})"
    )


def test_public_api_exists():
    """Verify all public API functions are importable."""
    from connes_cvs import (
        build_galerkin_matrix,
        compute_ground_state,
        extract_zeros,
        arb_eigenpair_residual_bound,
    )
    from connes_cvs.runner import CellConfig, GalerkinCell, run_cell
    from connes_cvs.sweep import run_sweep

    for fn in (build_galerkin_matrix, compute_ground_state, extract_zeros,
               arb_eigenpair_residual_bound, run_sweep,
               GalerkinCell, run_cell):
        assert callable(fn)
    assert CellConfig(c=13, N=8, T=60, dps=30).c == 13


def test_removed_unsafe_precision_apis_are_not_public():
    """Unproved precision heuristics must never reappear as guarantees."""
    import connes_cvs

    assert not hasattr(connes_cvs, "recommended_dps")
    assert not hasattr(connes_cvs, "precision_certificate")


def test_matching_digits_never_zero_pads():
    assert _matching_digits("1.230", "1.2300") == 4
    assert _matching_digits("1.2300", "1.230") == 4
    assert _matching_digits("2.865e-59", "2.865e-58") == 0
    assert _matching_digits("2.865e-59", "-2.865e-59") == 0


def test_karl_keysingularity_windows_artifact():
    """The credited v0.2.2 Windows artifact meets the quick-cell gate."""
    raw = KARL_ARTIFACT_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "5dddabbce30b4a4fa1a88f8ce34d82bb0d6b07e78801f5c36e54d7f7c428c79c"
    )
    artifact = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
    )
    reference = _load_reference("c13_quick_T400")
    assert artifact["schema"] == "rh-rig.groskin-c13-validation.v1"
    assert artifact["connes_cvs_version"] == "0.2.2"
    assert artifact["python"] == "3.11.9"
    assert artifact["parameters"] == {"c": 13, "N": 100, "T": 400, "dps": 80}
    assert _matching_digits(reference["lambda_min_even"], artifact["lambda_min"]) == 22
    assert _matching_digits(reference["gamma_1_abs_error"], artifact["gamma1_error"]) == 20


def test_arb_residual_bound_is_outward_and_scoped():
    flint = pytest.importorskip("flint")
    del flint
    import mpmath as mp
    from connes_cvs import arb_eigenpair_residual_bound

    mp.mp.dps = 80
    Q = mp.matrix([[0]])
    vector = mp.matrix([[1]])
    lam = mp.mpf(1) / 3
    result = arb_eigenpair_residual_bound(Q, vector, lam, dps=80)
    metadata = result["lambda_center_dyadic"]
    mantissa = int(metadata["mantissa_hex"], 16)
    if metadata["sign"] < 0:
        mantissa = -mantissa
    center = mp.mpf(
        (
            int(mantissa < 0),
            abs(mantissa),
            metadata["exponent_base_2"],
            abs(mantissa).bit_length(),
        )
    )
    assert center == lam
    assert mp.mpf(result["residual_bound"]) >= abs(center)
    assert result["rigorous_scope"] == "exact supplied finite symmetric mpf matrix only"


def test_arb_residual_bound_rejects_invalid_shapes_and_symmetry():
    pytest.importorskip("flint")
    import mpmath as mp
    from connes_cvs import arb_eigenpair_residual_bound

    with pytest.raises(ValueError, match="square"):
        arb_eigenpair_residual_bound(mp.matrix(1, 2), mp.matrix([[1]]), 0)
    with pytest.raises(ValueError, match="symmetric"):
        arb_eigenpair_residual_bound(
            mp.matrix([[1, 2], [0, 1]]), mp.matrix([[1], [0]]), 0
        )
    with pytest.raises(ValueError, match="nonzero"):
        arb_eigenpair_residual_bound(mp.eye(2), mp.matrix([[0], [0]]), 0)


# ============================================================
# Fast tests: extract_zeros v0.3.0 API (warning + c= parameter)
# ============================================================

def _tiny_eigvec():
    import mpmath as mp
    mp.mp.dps = 30
    return mp.matrix([[1], [mp.mpf("0.5")], [mp.mpf("0.25")], [mp.mpf("0.5")], [1]])


def test_extract_zeros_warns_on_float64_L():
    """A Python-float L must emit a UserWarning about the ~1e-16 cap."""
    import math
    from connes_cvs import extract_zeros

    v = _tiny_eigvec()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        with pytest.warns(UserWarning, match="float64"):
            extract_zeros(v, L=math.log(13), n_zeros=1, dps=30)


def test_extract_zeros_no_warning_on_mpf_L():
    """A full-precision mpf L must not warn."""
    import mpmath as mp
    from connes_cvs import extract_zeros

    v = _tiny_eigvec()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        extract_zeros(v, L=mp.log(13), n_zeros=1, dps=30)


def test_extract_zeros_accepts_c():
    """The c= parameter works, computes L internally, and does not warn."""
    from connes_cvs import extract_zeros

    v = _tiny_eigvec()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        results = extract_zeros(v, c=13, n_zeros=1, dps=30)
    assert len(results) == 1
    assert results[0]["k"] == 1


def test_extract_zeros_rejects_both_and_neither():
    """Passing both c and L, or neither, raises ValueError."""
    import mpmath as mp
    from connes_cvs import extract_zeros

    v = _tiny_eigvec()
    with pytest.raises(ValueError):
        extract_zeros(v, L=mp.log(13), c=13, n_zeros=1, dps=30)
    with pytest.raises(ValueError):
        extract_zeros(v, n_zeros=1, dps=30)


@pytest.mark.parametrize(
    "kwargs,exception",
    [
        ({"c": 1}, ValueError),
        ({"c": 13, "n_zeros": 0}, ValueError),
        ({"c": 13, "dps": 14}, ValueError),
        ({"c": 13, "tol": 0}, ValueError),
        ({"c": 13, "tol": "nan"}, ValueError),
        ({"c": 13.0}, TypeError),
    ],
)
def test_extract_zeros_validates_inputs(kwargs, exception):
    from connes_cvs import extract_zeros

    with pytest.raises(exception):
        extract_zeros(_tiny_eigvec(), **kwargs)


def test_build_and_ground_state_validate_shapes_and_types():
    import mpmath as mp
    from connes_cvs import build_galerkin_matrix, compute_ground_state

    with pytest.raises(TypeError):
        build_galerkin_matrix(13, N=2.5, T=20, dps=30)
    with pytest.raises(TypeError, match="Python float"):
        build_galerkin_matrix(13.0, N=1, T=20, dps=30)
    with pytest.raises(ValueError, match="odd dimension"):
        compute_ground_state(mp.eye(2))
    nonsymmetric = mp.eye(3)
    nonsymmetric[0, 1] = 1
    with pytest.raises(ValueError, match="symmetric"):
        compute_ground_state(nonsymmetric)
    complex_symmetric = mp.eye(3)
    complex_symmetric[0, 1] = complex_symmetric[1, 0] = mp.mpc(1, 1)
    with pytest.raises(ValueError, match="real"):
        compute_ground_state(complex_symmetric)


def test_rejected_cells_restore_the_process_precision_context():
    """
    A call rejected by validation must not change the global mpmath context.

    ``build_galerkin_matrix`` and ``extract_zeros`` raise the process-global
    ``mp.mp.dps`` to their ``dps`` argument *before* validating, so a rejected
    call used to leave that precision in place. Because
    ``compute_ground_state`` runs ``mpmath.eigsy`` at whatever precision is
    ambient and never sets one, a single ``TypeError`` from a typo (passing a
    Python float cutoff) silently changed ``lambda_min`` for every later matrix
    in the same process -- with no warning, on the quantity this suite pins to
    80 digits.
    """
    import mpmath as mp
    from connes_cvs import build_galerkin_matrix, compute_ground_state, extract_zeros

    Q = build_galerkin_matrix(c=13, N=2, T=60, dps=30)
    mp.mp.dps = 30
    lam_before, _ = compute_ground_state(Q)

    with pytest.raises(TypeError, match="Python float"):
        build_galerkin_matrix(13.0, N=2, T=60, dps=17)
    assert mp.mp.dps == 30
    assert compute_ground_state(Q)[0] == lam_before

    with pytest.raises(ValueError):
        build_galerkin_matrix(c=1, N=2, T=60, dps=17)
    assert mp.mp.dps == 30
    assert compute_ground_state(Q)[0] == lam_before

    _, eigvec = compute_ground_state(Q)
    with pytest.raises(ValueError):
        extract_zeros(eigvec, c=13, n_zeros=0, dps=17)
    assert mp.mp.dps == 30
    assert compute_ground_state(Q)[0] == lam_before

    with pytest.raises(ValueError):
        extract_zeros(eigvec, n_zeros=1, dps=17)
    assert mp.mp.dps == 30


def test_successful_cells_still_leave_their_precision_active():
    """
    The success path is unchanged: the builder leaves ``dps`` active.

    ``compute_ground_state`` relies on this and never sets a precision itself,
    so restoring on success would be a regression, not a hardening.
    """
    import mpmath as mp
    from connes_cvs import build_galerkin_matrix

    mp.mp.dps = 30
    build_galerkin_matrix(c=13, N=2, T=60, dps=42)
    assert mp.mp.dps == 42


def test_rejected_cells_restore_the_exact_binary_precision(monkeypatch):
    """
    Restoration must be exact in bits, not in decimal digits.

    mpmath is binary-precision based and its prec -> dps -> prec round trip is
    lossy: prec=101 reports dps=29, and assigning that dps back yields prec=100.
    A caller whose context was an awkward bit count therefore had its precision
    silently changed by the restore. Reported on PR #6.

    Starts from prec=101 so a dps-based restore is actually observable, and also
    drives a failure raised *after* the Flint context is configured, which is
    the other path that escapes the context manager.
    """
    import mpmath as mp
    from connes_cvs import build_galerkin_matrix, extract_zeros
    from connes_cvs import operator

    original_prec = mp.mp.prec
    original_flint = (
        (operator.flint_ctx.prec, operator.flint_ctx.threads)
        if operator.HAS_FLINT
        else None
    )

    def arm():
        """Put the ambient context on the awkward precision, then check it took."""
        mp.mp.prec = 101
        assert mp.mp.dps == 29, "prec=101 must be the lossy case this test guards"

    def assert_restored():
        assert mp.mp.prec == 101
        if operator.HAS_FLINT:
            assert (operator.flint_ctx.prec, operator.flint_ctx.threads) == original_flint

    def boom(*args, **kwargs):
        raise RuntimeError("injected after backend setup")

    try:
        arm()
        assert_restored()

        # Rejected during input validation.
        with pytest.raises(TypeError, match="Python float"):
            build_galerkin_matrix(13.0, N=2, T=60, dps=17)
        assert_restored()

        vector = _tiny_eigvec()  # the helper itself moves the global precision
        arm()
        with pytest.raises(ValueError):
            extract_zeros(vector, n_zeros=1, dps=17)
        assert_restored()

        # Raised after the backend context has been configured, not during
        # validation: the flint precision is already installed at this point.
        arm()
        monkeypatch.setattr(operator, "_compute_psi_pair", boom)
        with pytest.raises(RuntimeError, match="injected after backend setup"):
            build_galerkin_matrix(c=13, N=2, T=60, dps=17)
        assert_restored()

        # And the success path still leaves the requested precision active.
        monkeypatch.undo()
        arm()
        build_galerkin_matrix(c=13, N=2, T=60, dps=17)
        assert mp.mp.dps == 17
    finally:
        mp.mp.prec = original_prec
        if operator.HAS_FLINT:
            operator.flint_ctx.prec, operator.flint_ctx.threads = original_flint


def test_extract_zeros_rejects_zero_and_complex_vectors():
    import mpmath as mp
    from connes_cvs import extract_zeros

    with pytest.raises(ValueError, match="nonzero"):
        extract_zeros(mp.matrix(3, 1), c=13, n_zeros=1, dps=30)
    vector = mp.matrix([[1], [mp.mpc(1, 1)], [1]])
    with pytest.raises(ValueError, match="real"):
        extract_zeros(vector, c=13, n_zeros=1, dps=30)
    with pytest.raises(ValueError, match="reversal-even"):
        extract_zeros(mp.matrix([[1], [2], [3]]), c=13, n_zeros=1, dps=30)


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        (mp.nan, "must be finite"),
        (mp.inf, "must be finite"),
        (mp.mpc(14, 1), "finite real"),
        (mp.mpf(99), "local search window"),
    ],
)
def test_extract_zeros_rejects_invalid_root_candidates_and_clears_fields(
    monkeypatch, candidate, message
):
    import mpmath as mp
    from connes_cvs import extract_zeros

    monkeypatch.setattr(mp, "findroot", lambda *args, **kwargs: candidate)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        entry = extract_zeros(_tiny_eigvec(), c=13, n_zeros=1, dps=30)[0]
    assert entry["converged"] is False
    assert entry["gamma_detected"] is None
    assert entry["error"] is None
    assert entry["residual"] is None
    assert message in entry["failure"]


def test_extract_zeros_rejects_nonfinite_residual_and_clears_fields(monkeypatch):
    import mpmath as mp
    from connes_cvs import extract_zeros

    gamma = mp.im(mp.zetazero(1))
    monkeypatch.setattr(mp, "findroot", lambda *args, **kwargs: gamma)
    monkeypatch.setattr(mp, "exp", lambda *args, **kwargs: mp.mpc(mp.nan, mp.nan))
    with pytest.warns(RuntimeWarning, match="did not converge"):
        entry = extract_zeros(_tiny_eigvec(), c=13, n_zeros=1, dps=30)[0]
    assert entry["converged"] is False
    assert entry["gamma_detected"] is entry["error"] is entry["residual"] is None
    assert "residual must be finite" in entry["failure"]


def test_extract_zeros_residual_rejection_is_fail_closed_in_both_modes(monkeypatch):
    import mpmath as mp
    from connes_cvs import extract_zeros

    gamma = mp.im(mp.zetazero(1))
    monkeypatch.setattr(mp, "findroot", lambda *args, **kwargs: gamma)
    with pytest.warns(RuntimeWarning, match="residual exceeds"):
        entry = extract_zeros(
            _tiny_eigvec(), c=13, n_zeros=1, dps=30, tol="1e-25"
        )[0]
    assert entry["converged"] is False
    assert entry["gamma_detected"] is entry["error"] is entry["residual"] is None
    with pytest.raises(ValueError, match="residual exceeds"):
        extract_zeros(
            _tiny_eigvec(), c=13, n_zeros=1, dps=30, tol="1e-25", strict=True
        )


# ============================================================
# Slow regression gate (the real reference contract)
# ============================================================

@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_c13_ab_cell_full_printed_digit_match():
    """
    Recompute the A/B benchmark cell (c=13, N=80, T=400, dps=80) and
    require the FULL stored match on lambda_even: sign, decimal exponent,
    and every one of the 80 committed reference digits.

    The reference was computed at dps = 80 itself, so a same-precision
    recompute is deterministic and there is no headroom argument for a
    partial-digit floor here.  The computed value is printed with exactly
    the digit count the reference carries
    (``lambda_min_even_nstr_digits``), so the rounding of the final digit
    is performed identically on both sides and cannot produce a spurious
    failure.
    """
    if not _has_flint():
        pytest.skip("requires python-flint for a practical runtime")
    import mpmath as mp
    from connes_cvs.runner import CellConfig, GalerkinCell

    ref = _load_reference("c13_ab_benchmark")
    ref_str = ref["lambda_min_even"]
    n_digits = ref["lambda_min_even_nstr_digits"]
    assert _significant_digits(ref_str) == n_digits, (
        "reference_values.json is self-inconsistent: "
        f"lambda_min_even carries {_significant_digits(ref_str)} digits, "
        f"lambda_min_even_nstr_digits says {n_digits}"
    )

    cell = GalerkinCell(
        CellConfig(c=ref["c"], N=ref["N"], T=ref["T"], dps=ref["dps"]),
    )
    cell.run()
    got = mp.nstr(cell.lambda_even, n_digits)
    md = _matching_digits(ref_str, got)
    assert md >= n_digits, (
        f"lambda_even matches only {md} of the {n_digits} committed digits "
        "(full match required for this same-precision reference)\n"
        f"  reference: {ref_str}\n"
        f"  computed:  {got}"
    )


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_c13_quick_cell_lambda_and_gamma1():
    """
    Recompute the quick cell of examples/basic_compute.py
    (c=13, N=100, T=400, dps=80): lambda_even must match the committed
    25-digit reference (a dps=150 value) to >= 20 digits (measured match
    at dps=80: exactly 22, the dps=80 arithmetic floor), and the gamma_1
    extraction error (via the v0.3.0 c= parameter) must match the
    committed 20-digit reference to >= 18 digits (measured: all 20).
    """
    if not _has_flint():
        pytest.skip("requires python-flint for a practical runtime")
    import mpmath as mp
    from connes_cvs import extract_zeros
    from connes_cvs.runner import CellConfig, GalerkinCell

    ref = _load_reference("c13_quick_T400")
    cell = GalerkinCell(
        CellConfig(c=ref["c"], N=ref["N"], T=ref["T"], dps=ref["dps"]),
    )
    cell.run()
    got = mp.nstr(cell.lambda_even, 85)
    md = _matching_digits(ref["lambda_min_even"], got)
    assert md >= 20, (
        f"lambda_even matches only {md} digits (>= 20 required)\n"
        f"  reference: {ref['lambda_min_even']}\n"
        f"  computed:  {got}"
    )

    zeros = extract_zeros(cell.eigvec_full, c=13, n_zeros=1, dps=ref["dps"])
    err = zeros[0]["error"]
    assert err is not None, "gamma_1 root detection failed"
    got_err = mp.nstr(err, 40)
    md_err = _matching_digits(ref["gamma_1_abs_error"], got_err)
    assert md_err >= 18, (
        f"gamma_1 error matches only {md_err} digits (>= 18 required)\n"
        f"  reference: {ref['gamma_1_abs_error']}\n"
        f"  computed:  {got_err}"
    )


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_c13_published_t800_runner_cell_and_gamma1():
    """Recompute the actual published T=800 cell on the recorded stack."""
    if not _has_flint():
        pytest.skip("requires python-flint for a practical runtime")
    import mpmath as mp
    from connes_cvs import extract_zeros
    from connes_cvs.runner import CellConfig, GalerkinCell

    ref = _load_reference("c13_reference_T800")
    cell = GalerkinCell(
        CellConfig(c=ref["c"], N=ref["N"], T=ref["T"], dps=ref["dps"]),
        ground_state="minimum",
    )
    artifact = cell.run()
    regression = artifact["c13_regression"]
    assert regression["passed"] is True
    assert regression["matching_digits"] >= regression["threshold"] == 22
    got = mp.nstr(cell.lambda_even, 45)
    assert _matching_digits(ref["lambda_min_even"], got) >= 22

    zero = extract_zeros(
        cell.eigvec_full,
        c=ref["c"],
        n_zeros=1,
        dps=ref["dps"],
        strict=True,
    )[0]
    assert zero["converged"] is True
    got_error = mp.nstr(zero["error"], 40)
    assert _matching_digits(ref["gamma_1_abs_error"], got_error) >= 22
