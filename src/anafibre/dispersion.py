"""
dispersion.py

Contains the dispersion relation solver and associated utility functions for step-index fibers.
Implements normalized propagation constant (b), effective index (neff), and propagation constant (kz).
"""

import numpy as np

from scipy.special import jv, jvp, kv, kvp, kve
from scipy.optimize import root_scalar, root
from .utils import units, _HAS_UNITS, _strip_unit

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
        wl = fibre.wavelength_from_V(V)
        Vnum = np.asarray(V)
    elif wavelength is not None:
        wl = wavelength
        Vnum = fibre.V(wavelength)
    else:
        raise ValueError("Specify either V or wavelength.")
    
    #replace Vnum=0 with smallest number
    # Vnum[Vnum == 0] = np.finfo(float).eps

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
    # DlnK = -(kve(ell - 1, w) + kve(ell+1, w))/(2*kve(ell, w))

    # def wDlnK_safe_where(ell, w):
    #     # w = np.asarray(w)
    #     ell = abs(int(ell))  # K_{-nu} = K_{nu}
    #     with np.errstate(divide='ignore', invalid='ignore'):
    #         num = -w * kve(ell-1, w) 
    #         den = kve(ell, w)

    #         out = np.full_like(num, np.nan)

    #         np.divide(num, den, out=out,
    #                 where=np.isfinite(num) & np.isfinite(den) & (den != 0))
    #         out -= ell
    #         # out[np.isinf(out)] = -ell - (w[np.isinf(out)]**2) / (2.0*(ell-1))
    #         if ell >= 2:
    #             out[np.isnan(out)] = -ell - (w[np.isnan(out)]**2) / (2.0*(ell-1))
    #         elif ell == 1:
    #             out[np.isnan(out)] = -1.0 + 0.5 * (w[np.isnan(out)]**2) * (np.log(w[np.isnan(out)]/2.0) + np.euler_gamma - 0.5)
    #         else:
    #             out[np.isnan(out)] = 1.0 / (np.log(2.0/w[np.isnan(out)]) + np.euler_gamma)
    #         wtol = np.finfo(float).eps
    #         out[w < wtol] = np.nan
    #         return out

    wDlnK = _wDlnK(ell, w)
    wDlnK[w < np.finfo(float).eps] = np.nan
    
    phi_epsJ = eps1 * (u * b * Jp) + eps2 * ((1-b) * wDlnK * J)
    phi_muJ = mu1 * (u * b * Jp) + mu2 * ((1-b) * wDlnK * J)

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



def find_b_of_V(fibre, ell, m, V=None, wavelength=None, mode_type=None, N_b=2000, tol=np.nextafter(0, 1), complex_tol=1e-8, maxiter=100):
    if V is not None:
        arr = np.atleast_1d(V)
    elif wavelength is not None:
        wl = _strip_unit(wavelength, units.m if _HAS_UNITS else None)
        arr = np.atleast_1d(wl)
    else:
        raise ValueError("Specify either V or wavelength.")

    # V_arr = np.atleast_1d(V)
    out = np.full_like(arr, np.nan, dtype=float)
    bs = np.linspace(0, 1, N_b)

    if ell != 0 and mode_type is not None:
        mt = str(mode_type).strip().lower()
        if mt == "he":
            # m_in is radial index n → odd root index
            m = 2*int(m) - 1
        if mt == "eh":
            # m_in is radial index n → even root index
            m = 2*int(m)

    for i, arri in enumerate(arr): 
        if V is not None:
            def Ffun(bb):
                return F_dispersion(fibre, ell=ell, b=bb, V=arri, mode_type=mode_type)
        elif wavelength is not None:
            def Ffun(bb):
                return F_dispersion(fibre, ell=ell, b=bb, wavelength=arri, mode_type=mode_type)
        else:
            raise ValueError("Specify either V or wavelength.")

        Fvals = Ffun(bs)
        is_real = np.all(np.isreal(Fvals)) and np.all(np.abs(np.imag(Fvals)) < 1e-12)
        idx = np.where(np.sign(np.real(Fvals[:-1])) * np.sign(np.real(Fvals[1:])) < 0)[0]
        roots = []

        for j in idx:
            a, b_hi = bs[j], bs[j + 1]
            real_root = None
            try:
                sol_real = root_scalar(lambda bb: np.real(Ffun(bb)), bracket=[a, b_hi], method='brentq', xtol=tol)
                if sol_real.converged and a <= sol_real.root <= b_hi:
                    real_root = sol_real.root
            except Exception:
                pass

            if is_real and real_root is not None:
                roots.append(real_root)
            elif real_root is not None:
                def real_obj(bb_arr):
                    bb_scalar = float(np.atleast_1d(bb_arr).flatten()[0])
                    val = Ffun(bb_scalar)
                    return [np.real(val), np.imag(val)]

                try:
                    sol = root(real_obj, [real_root], method='hybr', tol=complex_tol, options={'maxfev': maxiter})
                    bb_sol = sol.x[0]
                    if sol.success and (a <= bb_sol <= b_hi):
                        val = Ffun(bb_sol)
                        if np.abs(val) < complex_tol:
                            roots.append(bb_sol)
                except Exception:
                    continue
            else:
                bb0 = 0.5 * (a + b_hi)
                def real_obj(bb_arr):
                    bb_scalar = float(np.atleast_1d(bb_arr).flatten()[0])
                    val = Ffun(bb_scalar)
                    return [np.real(val), np.imag(val)]

                try:
                    sol = root(real_obj, [bb0], method='hybr', tol=complex_tol, options={'maxfev': maxiter})
                    bb_sol = sol.x[0]
                    if sol.success and (a <= bb_sol <= b_hi):
                        val = Ffun(bb_sol)
                        if np.abs(val) < complex_tol:
                            roots.append(bb_sol)
                except Exception:
                    continue

        roots_sorted = sorted(roots, reverse=True)
        index = m - 1  # 1-based indexing!
        if 0 <= index < len(roots_sorted):
            out[i] = roots_sorted[index]

    return out[0] if np.isscalar(V) else out


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
