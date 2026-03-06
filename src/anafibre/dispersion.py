"""
Contains the dispersion relation solver and associated utility functions for step-index fibers.
Implements normalised propagation constant ($b$), effective index ($n_\\mathrm{e}$), and propagation constant ($k_z$).
"""

import numpy as np
from scipy.special import jv, jvp, kve
from scipy.optimize import root_scalar, root
from .utils import units, _HAS_UNITS, _strip_unit


def _resolve_wl_and_V(fibre, V=None, wl=None):
    if V is not None:
        try:
            wl_resolved = fibre.wavelength_from_V(V)
        except Exception:
            if hasattr(fibre, "wavelength_from_V_legacy"):
                wl_resolved = fibre.wavelength_from_V_legacy(V)
            else:
                raise
        return wl_resolved, np.asarray(V)
    if wl is not None:
        return wl, np.asarray(fibre.V(wl))
    raise ValueError("Specify either V or wavelength.")

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
    ell = abs(int(ell))  # K_{-nu} = K_{nu}
    w = np.asarray(w)

    w_nan_in = np.isnan(w)          # NaNs provided by caller
    w_finite = np.isfinite(w)

    with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
        num = -w * kve(ell - 1, w)
        den = kve(ell, w)

        out = np.full_like(w, np.nan, dtype=np.result_type(w, float))

        good = w_finite & np.isfinite(num) & np.isfinite(den) & (den != 0)
        np.divide(num, den, out=out, where=good)
        out -= ell

        # Only patch NaNs where input w was finite (so not user-NaNs)
        # Option A: patch all remaining NaNs on finite w
        patch = np.isnan(out) & w_finite

        # Option B (safer): patch only where w is actually small
        # patch = np.isnan(out) & w_finite & (np.abs(w) < np.finfo(float).eps)

        if np.any(patch):
            wp = w[patch]
            if ell >= 2:
                out[patch] = -ell - (wp**2) / (2.0*(ell - 1))
            elif ell == 1:
                out[patch] = -1.0 + 0.5 * (wp**2) * (np.log(wp/2.0) + np.euler_gamma - 0.5)
            else:
                out[patch] = 1.0 / (np.log(2.0/wp) + np.euler_gamma)

        # Restore caller-provided NaNs explicitly (belt-and-braces)
        out[w_nan_in] = np.nan

    return out

def Phi_alpha(ell, b, V, alpha=None):
    """Evaluate analytical function $\\Phi_\\ell^\\alpha(b,V)$.

    Parameters
    ----------
    ell : int
        Azimuthal mode index.
    b : float | complex | array-like
        Normalised propagation constant candidate(s).
    V : float | array-like
        Normalised frequency.
    alpha : sequence
        Length-2 sequence ``[alpha1, alpha2]`` defining core/cladding values.
        Each entry may be a scalar, array broadcastable with ``b``/``V``, or a
        callable ``f(V)``.

    Returns
    -------
    numpy.ndarray | float | complex
        Value(s) of $\\Phi_\\ell^\\alpha(b,V)$ with broadcasted input shape.

    Notes
    -----
    The implementation of the material-weighted logarithmic Bessel derivative follows the formulation:

    $\\Phi_\\ell^\\alpha(b,V)=\\left(\\dfrac{uw}{V}\\right)^2\\left[\\dfrac{\\alpha_1}{u}\\dfrac{\\mathrm{J}_\\ell'(u)}{\\mathrm{J}_\\ell(u)}+\\dfrac{\\alpha_2}{w}\\dfrac{\\mathrm{K}_\\ell'(w)}{\\mathrm{K}_\\ell(w)}\\right]$

    where $u=V\\sqrt{1-b}$ and $w=V\\sqrt{b}$.
    """
    
    b = np.asarray(b)
    if V is None:
        raise ValueError("Specify V.")
    Vnum = np.asarray(V)

    try:
        a1, a2 = alpha
    except Exception as exc:
        raise ValueError("alpha must be a length-2 sequence [alpha1, alpha2].") from exc
    if callable(a1):
        a1 = a1(Vnum)
    if callable(a2):
        a2 = a2(Vnum)

    sqb = np.sqrt(b)
    sqb1 = np.sqrt(1 - b)
    u = Vnum * sqb1
    w = Vnum * sqb
    J = jv(ell, u)
    Jp = jvp(ell, u)
    with np.errstate(divide="ignore", invalid="ignore"):
        DlnJ = Jp / J
        DlnK = _wDlnK(ell, w) / w
        pref = (u * w / Vnum) ** 2
        out = pref * ((a1 / u) * DlnJ + (a2 / w) * DlnK)
    return out

def F_dispersion(fibre, ell, b, V=None, wl=None, mode_type=None):
    """Evaluate analytical function $F_\\ell(b,V)$.

    Parameters
    ----------
    fibre : anafibre.fibre.StepIndexFibre
        Fibre model (material and geometry).
    ell : int
        Azimuthal mode index.
    b : float | complex | array-like
        Normalised propagation constant candidate(s).
    V : float | array-like, optional
        Normalised frequency. Provide either ``V`` or ``wl``.
    wl : float | array-like, optional
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
        If neither ``V`` nor ``wl`` is provided, or ``mode_type`` is invalid
        for ``ell == 0``.

    Notes
    -----
    The implementation follows the reduced dispersion formulation from the paper:

    $F_\\ell(b,V)=\\left[\\Phi_\\ell^\\varepsilon\\,\\Phi_\\ell^\\mu-(\\ell n_\\mathrm{e})^2\\right]\\mathrm{J}_\\ell^2(u)$

    where $n_\\mathrm{e}=\\sqrt{b\\,n_1^2+(1-b)\\,n_2^2}$, $u=V\\sqrt{1-b}$,
    and $w=V\\sqrt{b}$.

    For $\\ell=0$, TE/TM branches ($\\Phi_0^\\mu=0$/$\\Phi_0^\\varepsilon=0$) are selected by ``mode_type``; for $\\ell\\neq 0$
    the hybrid-mode scalar dispersion residual is used.
    """
    b = np.asarray(b)

    wl, Vnum = _resolve_wl_and_V(fibre, V=V, wl=wl)
    
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
    phi_eps = Phi_alpha(
        ell=ell, b=b, V=Vnum,
        alpha=[eps1, eps2],
    )
    phi_mu = Phi_alpha(
        ell=ell, b=b, V=Vnum,
        alpha=[mu1, mu2],
    )
    phi_epsJ = J * phi_eps
    phi_muJ = J * phi_mu

    if ell == 0:
        if mode_type is not None:
            mt = mode_type.lower()
            if mt == "te":
                return phi_muJ*sqb*sqb1
            elif mt == "tm":
                return phi_epsJ*sqb*sqb1
            else:
                raise ValueError("mode_type must be 'TE', 'TM', or None for ell=0.")
        else:
            return (phi_epsJ * phi_muJ - J ** 2 * (ell * ne) ** 2)*(b*(1-b))
    else:
        return (phi_epsJ * phi_muJ - J ** 2 * (ell * ne) ** 2)/ (ell * ell)*(b*(1-b))

def find_b_of_V(fibre, ell, m, V=None, wl=None, mode_type=None, N_b=2000, tol=np.nextafter(0, 1), complex_tol=1e-8, maxiter=200, return_complex=False):
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
        normalised frequency values. Provide either ``V`` or ``wl``.
    wl : float | array-like, optional
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
        If neither ``V`` nor ``wl`` is provided.

    Notes
    -----
    Root selection follows the reduced dispersion residual ``F_dispersion`` at each
    input sample. When the sampled residual on the real axis is effectively real,
    sign changes are bracketed and solved with Brent's method; otherwise complex
    roots are refined from minima of $|F|$ using a 2D nonlinear solve.

    Accepted roots are deduplicated, sorted by descending real part of $b$, and
    the returned mode is the $m$-th entry in that ordering (high to low
    $n_\\mathrm{e}$ at fixed $\\ell$ and $V$). If fewer than $m$ roots are found,
    the function returns ``nan`` (or ``nan+0j`` when complex output is requested).
    """
    scalar_input = np.isscalar(V) or np.isscalar(wl)

    if V is not None:
        arr = np.atleast_1d(V)
    elif wl is not None:
        wl = _strip_unit(wl, units.m if _HAS_UNITS else None)
        arr = np.atleast_1d(wl)
    else:
        raise ValueError("Specify either V or wavelength.")

    any_complex = False
    out = np.full(arr.shape, np.nan + 0j, dtype=complex)
    # bs = np.linspace(1e-9, 1 - 1e-9, N_b)
    bs = np.linspace(10*np.finfo(float).eps, 1, N_b)

    for i, arri in enumerate(arr):
        Ffun = (lambda bb: F_dispersion(fibre, ell=ell, b=bb, V=arri, mode_type=mode_type)) \
               if V is not None else \
               (lambda bb: F_dispersion(fibre, ell=ell, b=bb, wl=arri, mode_type=mode_type))

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

def b_to_neff(fibre, b, wl):
    """Convert normalised propagation constant ``b`` to effective index.

    Parameters
    ----------
    fibre : anafibre.fibre.StepIndexFibre
        Fibre model.
    b : float | array-like
        normalised propagation constant.
    wl : float | array-like
        Wavelength in meters.

    Returns
    -------
    float | numpy.ndarray
        Effective refractive index ``n_eff``.

    Notes
    -----
    Uses:

    $n_\\mathrm{e}=\\sqrt{b\\,n_1^2+(1-b)\\,n_2^2}$.
    """
    n1 = fibre.n_core(wl)
    n2 = fibre.n_clad(wl)
    b = np.asarray(b)
    return np.sqrt(b * n1**2 + (1 - b) * n2**2)

def b_to_kz(fibre, b, wl):
    """Convert normalised propagation constant ``b`` to longitudinal wavevector.

    Parameters
    ----------
    fibre : anafibre.fibre.StepIndexFibre
        Fibre model.
    b : float | array-like
        normalised propagation constant.
    wl : float | array-like
        Wavelength in meters.

    Returns
    -------
    float | numpy.ndarray
        Longitudinal propagation constant ``k_z`` in rad/m.

    Notes
    -----
    Uses: 
    
    $k_z=\\sqrt{b\\,k_1^2+(1-b)\\,k_2^2}$.
    """
    n_eff = b_to_neff(fibre, b, wl)
    wl = _strip_unit(wl, unit=units.m if _HAS_UNITS else None)
    k0 = 2 * np.pi / wl
    return n_eff * k0
