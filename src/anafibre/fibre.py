"""
fibre.py

Defines the StepIndexFibre class for cylindrical step-index optical fibers.
This class includes methods to compute refractive indices, normalized frequency V,
normalized propagation constant b, effective index neff, propagation constant kz,
and electromagnetic fields.

Author: Sebastian Golat
"""

import numpy as np
from .dispersion import find_b_of_V, b_to_neff, b_to_kz, F_dispersion
from .fields import GuidedMode
# from .normalization import compute_power, compute_normalisation
from .utils import units, _HAS_UNITS, _strip_unit
from scipy.optimize import brentq
# import refractiveindex
from refractiveindex.refractiveindex  import RefractiveIndex, NoExtinctionCoefficient


        
class RefractiveIndexMaterial:
    def __init__(self, shelf, book, page, **ri_kwargs):
        BD = RefractiveIndex(**ri_kwargs)
        # Material(self.getMaterialFilename(shelf, book, page))
        self.material = BD.getMaterial(shelf=shelf, book=book, page=page)
        self.rangeMin = self.material.refractiveIndex.rangeMin
        self.rangeMax = self.material.refractiveIndex.rangeMax
        
        if np.isnan(self.get_eps(self.rangeMin*1e-6)):
            self.rangeMin += 0.1
        if np.isnan(self.get_eps(self.rangeMax*1e-6)):
            self.rangeMax -= 0.1
        # print(self.rangeMax, self.rangeMin)
        # print(self.get_eps(self.rangeMin*1e-6), self.get_eps(self.rangeMax*1e-6))

    def get_refractive_index(self, wl):
        return self.material.getRefractiveIndex(np.copy(wl), bounds_error=False)
    
    def get_extinction_coefficient(self, wl):
        return self.material.getExtinctionCoefficient(np.copy(wl), bounds_error=False)

    def get_eps(self, wavelength, exp_type='exp_minus_i_omega_t', real=True):
        wl = _strip_unit(wavelength, units.m if _HAS_UNITS else None)*1e9  # Convert to nm if using astropy units
        n = self.get_refractive_index(wl)
        if real:
            return n**2
        try:
            k = self.get_extinction_coefficient(wl)
            if exp_type=='exp_minus_i_omega_t':
                return (n + 1j*k)**2
            else:
                return (n - 1j*k)**2
        except NoExtinctionCoefficient:
            return n**2


class StepIndexFibre:
    def __init__(
        self,
        core_radius,
        *,
        # Highest precedence: materials
        core: "RefractiveIndexMaterial | None" = None,
        clad: "RefractiveIndexMaterial | None" = None,
        # Next: direct permittivity
        eps_core=None,
        eps_clad=None,
        # Last: refractive index
        n_core=None,
        n_clad=None,
        # Permeability (can be scalar or callable λ→μ)
        mu_core=1.0,
        mu_clad=1.0,
        # When using RefractiveIndexMaterial, control ε(λ) form
        exp_type: str = "exp_minus_i_omega_t",
        real_eps_from_material: bool = True,
    ):
        """
        Parameters
        ----------
        core_radius : float or Quantity[m]
        core, clad  : RefractiveIndexMaterial (highest precedence, per side)
        eps_*       : scalar or callable λ→ε(λ) (used if material not provided)
        n_*         : scalar or callable λ→n(λ)  (used if neither material nor ε_* provided)
        mu_*        : scalar or callable λ→μ(λ)
        exp_type    : passed to RefractiveIndexMaterial.get_eps when core/clad given
        real_eps_from_material : if True, use n^2 (real). If False and k is available in
                                 the database, use (n ± i k)^2 depending on exp_type.
        """

        self.core_radius = _strip_unit(core_radius, units.m if _HAS_UNITS else None)
        self.core = core
        self.clad = clad

        # Store μ (can be scalar or callable)
        self.mu_core = mu_core
        self.mu_clad = mu_clad

        # Helpers to wrap constants/callables uniformly
        def _as_callable(val):
            if callable(val):
                return val
            return (lambda wl, _v=val: _v)

        def _eps_from_n(n_val, mu_val):
            n_fun = _as_callable(n_val)
            mu_fun = _as_callable(mu_val)
            return lambda wl: (n_fun(wl) ** 2) / mu_fun(wl)

        def _eps_from_material(mat: "RefractiveIndexMaterial"):
            # Use database ε(λ). Choose real n^2 or complex (n ± i k)^2.
            if real_eps_from_material:
                return lambda wl, m=mat: m.get_eps(wl, real=True)
            else:
                return lambda wl, m=mat: m.get_eps(wl, exp_type=exp_type, real=False)

        # Per-side resolution with precedence: material → ε → n
        def _pick_eps(side, mat, eps_val, n_val, mu_val):
            if mat is not None:
                return _eps_from_material(mat)
            if eps_val is not None:
                return _as_callable(eps_val)
            if n_val is not None:
                return _eps_from_n(n_val, mu_val)
            raise ValueError(
                f"Insufficient data for {side}: provide either {side} material, "
                f"or eps_{side}, or n_{side}."
            )

        self.eps_core = _pick_eps("core", core, eps_core, n_core, self.mu_core)
        self.eps_clad = _pick_eps("clad", clad, eps_clad, n_clad, self.mu_clad)


    def _eval(self, val, wavelength):
        if callable(val):
            return val(wavelength)
        return val

    def n_core(self, wavelength):
        wl = _strip_unit(wavelength, units.m if _HAS_UNITS else None)
        eps = self._eval(self.eps_core, wl)
        mu = self._eval(self.mu_core, wl)
        return np.sqrt(eps * mu)

    def n_clad(self, wavelength):
        wl = _strip_unit(wavelength, units.m if _HAS_UNITS else None)
        eps = self._eval(self.eps_clad, wl)
        mu = self._eval(self.mu_clad, wl)
        return np.sqrt(eps * mu)

    def eps(self, r, wavelength):
        rr = _strip_unit(r, units.m if _HAS_UNITS else None)
        wl = _strip_unit(wavelength, units.m if _HAS_UNITS else None)
        return np.where(rr < self.core_radius,
                        self._eval(self.eps_core, wl),
                        self._eval(self.eps_clad, wl))

    def mu(self, r, wavelength):
        rr = _strip_unit(r, units.m if _HAS_UNITS else None)
        wl = _strip_unit(wavelength, units.m if _HAS_UNITS else None)
        return np.where(rr < self.core_radius,
                        self._eval(self.mu_core, wl),
                        self._eval(self.mu_clad, wl))

    def n(self, r, wavelength):
        return np.sqrt(self.eps(r, wavelength) * self.mu(r, wavelength))

    def V(self, wavelength):
        wl = _strip_unit(wavelength, units.m if _HAS_UNITS else None)
        n1 = self.n_core(wl)
        n2 = self.n_clad(wl)
        k0 = 2 * np.pi / wl
        return k0 * self.core_radius * np.sqrt(n1**2 - n2**2)

    # def wavelength_from_V(self, V):
    #     # raise ValueError("Ha! This method is not implemented in StepIndexFibre. Use a different method to compute wavelength from V.")
    #     n1 = self.n_core(1.0)
    #     n2 = self.n_clad(1.0)
        # a = self.core_radius
        # delta_n2 = n1**2 - n2**2
        # if delta_n2 <= 0:
        #     raise ValueError("Core index must be larger than cladding index.")
        # V = np.asarray(V)
        # result = np.empty_like(V, dtype=float)
        # result[V == 0] = np.inf
        # result[V != 0] = 2 * np.pi * a * np.sqrt(delta_n2) / V[V != 0]
        # return result if result.shape != () else result.item()

    def wavelength_from_V(
        self,
        V,
        wl_bracket=(2e-7, 5e-6),  # meters: search range if indices are callable
        rtol=1e-12,
        maxiter=100,
        _grid_pts=64,
    ):
        """
        Invert V(λ) = (2π a / λ) * sqrt(n_core(λ)^2 - n_clad(λ)^2) for λ.

        Works when eps_core/mu_core/eps_clad/mu_clad are constants or callables of wavelength [m].
        Returns np.inf for V==0. Vectorized over V.
        """
        a = self.core_radius
        V = np.asarray(V)

        # --- Correct constant-index detection (attributes, not methods) ---
        indices_constant = (
            not callable(self.eps_core) and
            not callable(self.mu_core) and
            not callable(self.eps_clad) and
            not callable(self.mu_clad)
        )

        # Fast closed-form path if both indices are constants
        if indices_constant:
            # It’s safe to evaluate at any λ because they’re constants
            n1 = float(self.n_core(1.0))
            n2 = float(self.n_clad(1.0))
            dn2 = n1*n1 - n2*n2
            if dn2 <= 0:
                raise ValueError("Core index must be larger than cladding index (n_core^2 > n_clad^2).")
            out = np.empty_like(V, dtype=float)
            mask0 = (V == 0)
            maskp = ~mask0
            out[mask0] = np.inf
            out[maskp] = 2*np.pi*a*np.sqrt(dn2) / V[maskp]
            return out if out.shape != () else out.item()


        # --- General path: wavelength-dependent indices ---
        wl_min, wl_max = wl_bracket
        if not (wl_min > 0 and wl_max > wl_min and np.isfinite([wl_min, wl_max]).all()):
            raise ValueError("wl_bracket must be finite positives with wl_max > wl_min.")

        def solve_one(V_target):
            if V_target < 0:
                raise ValueError("V must be nonnegative.")
            if V_target == 0:
                return np.inf

            lam_grid = np.geomspace(wl_min, wl_max, _grid_pts)
            # Evaluate V(λ) on the grid (vectorized)
            V_span = np.asarray([self.V(l) for l in lam_grid], dtype=float)
            fvals = V_span - V_target

            # exact hit?
            finite = np.isfinite(fvals)
            hit = np.where(finite & (fvals == 0))[0]
            if hit.size:
                return lam_grid[hit[0]]

            # look for a sign change
            left = right = None
            for i in range(len(lam_grid) - 1):
                f1, f2 = fvals[i], fvals[i+1]
                if not (np.isfinite(f1) and np.isfinite(f2)):
                    continue
                if (f1 < 0 and f2 > 0) or (f1 > 0 and f2 < 0):
                    left, right = lam_grid[i], lam_grid[i+1]
                    break

            if left is None:
                Vmin = np.nanmin(V_span)
                Vmax = np.nanmax(V_span)
                raise ValueError(
                    f"No λ in wl_bracket produces V={V_target:.6g}. "
                    f"Observed V(λ) in [{wl_min:.3g}, {wl_max:.3g}] m spans ≈ [{Vmin:.6g}, {Vmax:.6g}]. "
                    "Try expanding wl_bracket or check n_core/n_clad validity there."
                )

            f = lambda lam: float(self.V(lam) - V_target)
            return brentq(f, left, right, rtol=rtol, maxiter=maxiter)

        if V.shape == ():
            return solve_one(float(V))

        out = np.empty_like(V, dtype=float)
        it = np.nditer(V, flags=['multi_index'])
        for v in it:
            out[it.multi_index] = solve_one(float(v))
        return out

    def __repr__(self):
        return (f"StepIndexFibre(core_radius={self.core_radius:.3e}, "
                f"eps_core={self.eps_core}, eps_clad={self.eps_clad}, "
                f"mu_core={self.mu_core}, mu_clad={self.mu_clad})")

    def b(self, ell, m, V=None, wavelength=None, mode_type=None,
          N_b=2000, tol=1e-15, complex_tol=1e-8, maxiter=100):
        if V is not None:
            return find_b_of_V(self, ell, m, V=V, mode_type=mode_type, N_b=N_b, tol=tol, complex_tol=complex_tol, maxiter=maxiter)
        elif wavelength is not None:
            return find_b_of_V(self, ell, m, wavelength=wavelength, mode_type=mode_type, N_b=N_b, tol=tol, complex_tol=complex_tol, maxiter=maxiter)
        else:
            raise ValueError("Specify either V or wavelength.")
        

    def neff(self, ell, m, V=None, wavelength=None, mode_type=None, **kwargs):
        if wavelength is not None:
            wl = wavelength
        elif V is not None:
            wl = self.wavelength_from_V(V)
        else:
            raise ValueError("Specify either V or wavelength.")
        b_val = self.b(ell, m, V=V, wavelength=wavelength, mode_type=mode_type, **kwargs)
        return b_to_neff(self, b_val, wl)

    def kz(self, ell, m, V=None, wavelength=None, mode_type=None, **kwargs):
        if wavelength is not None:
            wl = wavelength
        elif V is not None:
            wl = self.wavelength_from_V(V)
        else:
            raise ValueError("Specify either V or wavelength.")
        b_val = self.b(ell, m, V=V, wavelength=wavelength, mode_type=mode_type, **kwargs)
        return b_to_kz(self, b_val, wl)
    
    def F(self, ell, b, V=None, wavelength=None, mode_type=None):
        """
        Return the dispersion function F(b, V) or F(b, wavelength) for given mode.

        Parameters
        ----------
        ell : int
            Azimuthal mode number.
        b : float or array-like
            Normalized propagation constant.
        V : float, optional
            Normalized frequency.
        wavelength : float, optional
            Wavelength (used if V not given).
        mode_type : str, optional
            For ell=0, specify 'TE' or 'TM'.

        Returns
        -------
        float or ndarray
            Value of the dispersion equation F(b, V).
        """
        return F_dispersion(self, ell, b, V=V, wavelength=wavelength, mode_type=mode_type)
    
    def m_max(self, ell, wavelength, mode_type=None, N_b=2000, tol=1e-15, complex_tol=1e-8, maxiter=100):
        """
        Return the maximum allowed radial mode number m for given ell and V.
        """
        m = 1
        while True:
            b = self.b(ell, m, wavelength=wavelength, mode_type=mode_type, N_b=N_b, tol=tol, complex_tol=complex_tol, maxiter=maxiter)
            # If b is nan, not a valid solution: break
            if np.isnan(b) or not (0 < b < 1):
                break
            m += 1
        return m - 1  # The last valid one
    
    def ell_max(self, wavelength, m=1, mode_type=None, N_b=2000, tol=1e-15, complex_tol=1e-8, maxiter=100, ell_max_search=100):
        """
        Return the maximum allowed azimuthal mode number ell for given V.
        By default, checks for fundamental radial mode (m=1) for increasing ell.
        ell_max_search is a safety limit to prevent infinite loops.
        """
        ell = 0
        while ell < ell_max_search:
            b = self.b(ell, m, wavelength=wavelength, mode_type=mode_type, N_b=N_b, tol=tol, complex_tol=complex_tol, maxiter=maxiter)
            if np.isnan(b) or not (0 < b < 1):
                break
            ell += 1
        return ell - 1  # Last valid ell
    def HE(self, ell, n, wl, **kwargs):
        """Return an HE mode (ell>0, odd m)."""
        return GuidedMode(self, ell=ell, m=n, wavelength=wl,
                          mode_type="HE", **kwargs)

    def EH(self, ell, n, wl, **kwargs):
        """Return an EH mode (ell>0, even m)."""
        return GuidedMode(self, ell=ell, m=n, wavelength=wl,
                          mode_type="EH", **kwargs)

    def TE(self, n, wl, **kwargs):
        """Return a TE₀m mode (ell=0)."""
        return GuidedMode(self, ell=0, m=n, wavelength=wl,
                          mode_type="TE", **kwargs)

    def TM(self, n, wl, **kwargs):
        """Return a TM₀m mode (ell=0)."""
        return GuidedMode(self, ell=0, m=n, wavelength=wl,
                          mode_type="TM", **kwargs)