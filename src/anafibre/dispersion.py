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
    n1 = fibre.n_core(wavelength)
    n2 = fibre.n_clad(wavelength)
    b = np.asarray(b)
    return np.sqrt(b * n1**2 + (1 - b) * n2**2)

def b_to_kz(fibre, b, wavelength):
    n_eff = b_to_neff(fibre, b, wavelength)
    wl = _strip_unit(wavelength, unit=units.m if _HAS_UNITS else None)
    k0 = 2 * np.pi / wl
    return n_eff * k0
