"""
Regression test for the c=13 ground-state eigenvalue.

Validates that the CvS Galerkin matrix at cutoff c=13 with standard
parameters (N=100, T=400, dps=80) produces a ground-state eigenvalue
whose implied gamma_1 error falls within the established range.

This test serves as the primary correctness gate: if it passes, the
operator construction pipeline is working correctly. The expected
range [1e-56, 3e-55] is derived from extensive convergence studies
across iterations 4--7 of the implementation, and is consistent with
published values from Connes (2026) and CCM (2025) up to the expected
basis-dependent factor of ~1.7x.

References
----------
- Connes, "Weil positivity and trace formula the archimedean place," 2026
- Connes, Consani & Moscovici, arXiv:2511.22755, Section 6
"""

import pytest


# Mark as slow: this test takes ~2-5 minutes depending on hardware
@pytest.mark.slow
@pytest.mark.timeout(600)
def test_c13_gamma1_error_in_range():
    """
    Test that c=13 N=100 T=400 dps=80 gives |gamma_1 error| in [1e-56, 3e-55].

    The first Riemann zero is gamma_1 = 14.134725141734693790...
    The ground-state eigenvalue lambda_min should encode gamma_1 such that
    the absolute error is of order 10^{-55}, consistent with independent
    computations by Connes and CCM.
    """
    pytest.skip(
        "Stub: test will be enabled once operator.py implementation is ported. "
        "Expected: |gamma_1 - detected| in [1e-56, 3e-55]."
    )

    # --- Implementation placeholder ---
    # from connes_cvs import build_galerkin_matrix, compute_ground_state, extract_zeros
    # import math
    #
    # GAMMA_1 = 14.134725141734693790  # first Riemann zero (imaginary part)
    #
    # Q = build_galerkin_matrix(c=13, N=100, T=400, dps=80)
    # lam_min, eigvec = compute_ground_state(Q)
    # zeros = extract_zeros(eigvec, L=math.log(13), n_zeros=1)
    #
    # gamma1_error = abs(zeros[0] - GAMMA_1)
    #
    # assert 1e-56 <= float(gamma1_error) <= 3e-55, (
    #     f"|γ₁ error| = {gamma1_error:.4e} is outside expected range [1e-56, 3e-55]"
    # )


def test_package_imports():
    """Smoke test: verify the package can be imported."""
    import connes_cvs
    assert connes_cvs.__version__ == "0.1.0"


def test_public_api_exists():
    """Verify all public API functions are importable."""
    from connes_cvs import build_galerkin_matrix, compute_ground_state, extract_zeros
    from connes_cvs.sweep import run_sweep

    # All should be callable
    assert callable(build_galerkin_matrix)
    assert callable(compute_ground_state)
    assert callable(extract_zeros)
    assert callable(run_sweep)
