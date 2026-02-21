"""
dispersion.py

Contains the dispersion relation solver and associated utility functions for step-index fibers.
Implements normalized propagation constant (b), effective index (neff), and propagation constant (kz).
"""

import numpy as np

from scipy.special import jv, jvp, kv, kvp, kve
from scipy.optimize import root_scalar, root
from .utils import units, _HAS_UNITS, _strip_unit

def _dedupe_complex_roots(roots, tol=1e-7):
    uniq = []
    for r in roots:
        if not any(abs(r - u) < tol for u in uniq):
            uniq.append(r)
    return uniq

def _candidate_seeds_from_absF(Ffun, bs, q=0.08):
    # local minima of |F| on real b-axis
    vals = np.array([abs(Ffun(float(b))) for b in bs], dtype=float)
    seeds = []
    for i in range(1, len(bs) - 1):
        if vals[i] <= vals[i - 1] and vals[i] <= vals[i + 1]:
            seeds.append(float(bs[i]))
    if not seeds:
        seeds = [float(bs[np.argmin(vals)])]
    # keep best fraction only
    seeds = sorted(seeds, key=lambda x: abs(Ffun(x)))
    keep = max(1, int(np.ceil(q * len(seeds))))
    return seeds[:keep]

def _wDlnK(ell, w):
        # w = np.asarray(w)
        ell = abs(int(ell))  # K_{-nu} = K_{nu}
        with np.errstate(divide='ignore', invalid='ignore'):
            num = -w * kve(ell-1, w) 
            den = kve(ell, w)
            out = np.full_like(w, np.nan)

            np.divide(num, den, out=out,
                    where=np.isfinite(num) & np.isfinite(den) & (den != 0))
            out -= ell
            # Apply the asymptotic expansion for small w to avoid divergence at w=0
            if ell >= 2:
                out[np.isnan(out)] = -ell - (w[np.isnan(out)]**2) / (2.0*(ell-1)) 
            elif ell == 1:
                out[np.isnan(out)] = -1.0 + 0.5 * (w[np.isnan(out)]**2) * (np.log(w[np.isnan(out)]/2.0) + np.euler_gamma - 0.5)
            else:
                out[np.isnan(out)] = 1.0 / (np.log(2.0/w[np.isnan(out)]) + np.euler_gamma)
            return out

def F_dispersion(fibre, ell, b, V=None, wavelength=None, mode_type=None):
    """Evaluate the step-index fibre dispersion function.

    Parameters
    ----------
    fibre : anafibre.fibre.StepIndexFibre
        Fibre model used to evaluate material and geometric terms.
    ell : int
        Azimuthal mode index.
    b : float | complex | array-like
        Normalized propagation constant candidate(s).
    V : float | array-like, optional
        Normalized frequency. Provide either ``V`` or ``wavelength``.
    wavelength : float | array-like, optional
        Wavelength in meters. Used when ``V`` is not provided.
    mode_type : {"TE", "TM", None}, optional
        For ``ell == 0``, selects TE/TM branch. Ignored for ``ell != 0``.

    Returns
    -------
    numpy.ndarray | float | complex
        Dispersion residual with the same broadcasted shape as the inputs.

    Raises
    ------
    ValueError
        If neither ``V`` nor ``wavelength`` is provided, or ``mode_type`` is invalid
        for ``ell == 0``.

    Notes
    -----
    The implementation follows the reduced dispersion formulation from the paper:

    - $n_\\mathrm{eff}(b)=\\sqrt{b\\,n_1^2+(1-b)\\,n_2^2}$
    - $u=V\\sqrt{1-b}$, $w=V\\sqrt{b}$
    - $F_\\ell$ is built from material-weighted logarithmic Bessel derivatives.

    For $\\ell=0$, TE/TM branches are selected by ``mode_type``; for $\\ell\\neq 0$
    the hybrid-mode scalar dispersion residual is used.
    """
    b = np.asarray(b)

    if V is not None:
        try:
            wl = fibre.wavelength_from_V(V)
        except:
            wl = fibre.wavelength_from_V_legacy(V)
        Vnum = np.asarray(V)
    elif wavelength is not None:
        wl = wavelength
        Vnum = fibre.V(wavelength)
    else:
        raise ValueError("Specify either V or wavelength.")
    
    n1 = fibre.n_core(wl)
    n2 = fibre.n_clad(wl)
    eps1 = fibre._eval(fibre.eps_core, wl)
    eps2 = fibre._eval(fibre.eps_clad, wl)
    mu1 = fibre._eval(fibre.mu_core, wl)
    mu2 = fibre._eval(fibre.mu_clad, wl)

    ne = np.sqrt(b * n1**2 + (1 - b) * n2**2)
    sqb = np.sqrt(b)
    sqb1 = np.sqrt(1 - b)
    u = Vnum * sqb1
    w = Vnum * sqb

    J = jv(ell, u)
    Jp = jvp(ell, u)

    wDlnK = _wDlnK(ell, w)
    wDlnK[np.abs(w) < np.finfo(float).eps] = np.nan
    
    phi_epsJ = eps1 * (u * b * Jp) + eps2 * ((1-b) * wDlnK * J)
    phi_muJ  = mu1  * (u * b * Jp) + mu2  * ((1-b) * wDlnK * J)

    if ell == 0:
        if mode_type is not None:
            mt = mode_type.lower()
            if mt == "te":
                return phi_epsJ*sqb*sqb1
            elif mt == "tm":
                return phi_muJ*sqb*sqb1
            else:
                raise ValueError("mode_type must be 'TE', 'TM', or None for ell=0.")
        else:
            return (phi_epsJ * phi_muJ - J ** 2 * (ell * ne) ** 2)*(b*(1-b))
    else:
        return (phi_epsJ * phi_muJ - J ** 2 * (ell * ne) ** 2)/ (ell * ell)*(b*(1-b))

def find_b_of_V(fibre, ell, m, V=None, wavelength=None, mode_type=None, N_b=2000, tol=np.nextafter(0, 1), complex_tol=1e-8, maxiter=200, return_complex=False):
    """Solve for the guided root ``b`` of a specific mode branch.

    Parameters
    ----------
    fibre : anafibre.fibre.StepIndexFibre
        Fibre model.
    ell : int
        Azimuthal mode index.
    m : int
        Radial mode index (1-based root ordering by descending real ``b``).
    V : float | array-like, optional
        Normalized frequency values. Provide either ``V`` or ``wavelength``.
    wavelength : float | array-like, optional
        Wavelength values in meters.
    mode_type : {"TE", "TM", None}, optional
        TE/TM selection for ``ell == 0``.
    N_b : int, default=2000
        Number of samples used for bracketing/seeding roots on the real axis.
    tol : float, default=np.nextafter(0, 1)
        Scalar root tolerance used by Brent's method.
    complex_tol : float, default=1e-8
        Residual tolerance for complex root acceptance.
    maxiter : int, default=200
        Maximum function evaluations for complex solver iterations.
    return_complex : bool, default=False
        If ``True``, keeps complex-valued roots. If ``False``, returns real values
        when roots are effectively real.

    Returns
    -------
    float | complex | numpy.ndarray
        Mode root ``b`` for each input sample. Missing roots are returned as ``nan``
        (or ``nan+0j`` when complex output is requested).

    Raises
    ------
    ValueError
        If neither ``V`` nor ``wavelength`` is provided.

    Notes
    -----
    Roots are selected as the $m$-th solution after sorting by descending real
    part of $b$ (consistent with the paper's high-$n_\\mathrm{eff}$ to low-$n_\\mathrm{eff}$ mode
    indexing at fixed $\\ell$ and $V$).
    """
    scalar_input = np.isscalar(V) or np.isscalar(wavelength)

    if V is not None:
        arr = np.atleast_1d(V)
    elif wavelength is not None:
        wl = _strip_unit(wavelength, units.m if _HAS_UNITS else None)
        arr = np.atleast_1d(wl)
    else:
        raise ValueError("Specify either V or wavelength.")

    any_complex = False
    out = np.full(arr.shape, np.nan + 0j, dtype=complex)
    bs = np.linspace(1e-9, 1 - 1e-9, N_b)

    for i, arri in enumerate(arr):
        Ffun = (lambda bb: F_dispersion(fibre, ell=ell, b=bb, V=arri, mode_type=mode_type)) \
               if V is not None else \
               (lambda bb: F_dispersion(fibre, ell=ell, b=bb, wavelength=arri, mode_type=mode_type))

        Fvals = Ffun(bs)
        effectively_real = np.all(np.abs(np.imag(Fvals)) < 1e-12)

        roots = []
        if effectively_real:
            idx = np.where(np.sign(np.real(Fvals[:-1])) * np.sign(np.real(Fvals[1:])) < 0)[0]
            for j in idx:
                a, b_hi = bs[j], bs[j + 1]
                try:
                    sol = root_scalar(lambda bb: np.real(Ffun(bb)), bracket=[a, b_hi], method="brentq", xtol=tol)
                    if sol.converged:
                        roots.append(complex(sol.root, 0.0))
                except Exception:
                    pass
        else:
            any_complex = True
            seeds = _candidate_seeds_from_absF(Ffun, bs)

            def sys(xy):
                b = xy[0] + 1j * xy[1]
                v = Ffun(b)
                return np.array([np.real(v), np.imag(v)], dtype=float)

            for br0 in seeds:
                try:
                    sol = root(sys, x0=np.array([br0, 0.0]), method="hybr",
                               tol=complex_tol, options={"maxfev": maxiter})
                    if not sol.success:
                        continue
                    bsol = sol.x[0] + 1j * sol.x[1]
                    if (0 < np.real(bsol) < 1) and (abs(Ffun(bsol)) < complex_tol):
                        roots.append(bsol)
                except Exception:
                    pass

        roots = _dedupe_complex_roots(roots, tol=1e-7)
        roots = sorted(roots, key=lambda z: np.real(z), reverse=True)
        idx_mode = m - 1
        if 0 <= idx_mode < len(roots):
            out[i] = roots[idx_mode]

    if scalar_input:
        val = out.flat[0]
        if (not return_complex) and (abs(np.imag(val)) < 1e-14):
            return float(np.real(val))
        return val

    if (not return_complex) and (not any_complex):
        return np.real(out)
    return out

def b_to_neff(fibre, b, wavelength):
    """Convert normalized propagation constant ``b`` to effective index.

    Parameters
    ----------
    fibre : anafibre.fibre.StepIndexFibre
        Fibre model.
    b : float | array-like
        Normalized propagation constant.
    wavelength : float | array-like
        Wavelength in meters.

    Returns
    -------
    float | numpy.ndarray
        Effective refractive index ``n_eff``.

    Notes
    -----
    Uses
    $n_\\mathrm{eff}=\\sqrt{b\\,n_1^2+(1-b)\\,n_2^2}$.
    """
    n1 = fibre.n_core(wavelength)
    n2 = fibre.n_clad(wavelength)
    b = np.asarray(b)
    return np.sqrt(b * n1**2 + (1 - b) * n2**2)

def b_to_kz(fibre, b, wavelength):
    """Convert normalized propagation constant ``b`` to longitudinal wavevector.

    Parameters
    ----------
    fibre : anafibre.fibre.StepIndexFibre
        Fibre model.
    b : float | array-like
        Normalized propagation constant.
    wavelength : float | array-like
        Wavelength in meters.

    Returns
    -------
    float | numpy.ndarray
        Longitudinal propagation constant ``k_z`` in rad/m.

    Notes
    -----
    Uses $k_z=n_\\mathrm{eff}k_0$ with $k_0=2\\pi/\\lambda$.
    """
    n_eff = b_to_neff(fibre, b, wavelength)
    wl = _strip_unit(wavelength, unit=units.m if _HAS_UNITS else None)
    k0 = 2 * np.pi / wl
    return n_eff * k0
