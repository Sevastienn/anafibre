"""
fibre.py

Defines the StepIndexFibre class for cylindrical step-index optical fibers.
This class includes methods to compute refractive indices, normalized frequency V,
normalized propagation constant b, effective index neff, propagation constant kz,
and electromagnetic fields.

Author: Sebastian Golat
"""

import numpy as np
import warnings

from .utils import repr_html_modes
from .dispersion import find_b_of_V, b_to_neff, b_to_kz, F_dispersion
from .fields import GuidedMode
# from .normalization import compute_power, compute_normalisation
from .utils import units, _HAS_UNITS, _strip_unit
from scipy.optimize import brentq

# Optional refractiveindex dependency
try:
    from refractiveindex.refractiveindex import RefractiveIndexMaterial as RIMaterial, NoExtinctionCoefficient
    _HAS_REFRACTIVEINDEX = True
except ImportError:
    _HAS_REFRACTIVEINDEX = False
    RIMaterial = None
    NoExtinctionCoefficient = Exception
        
class RefractiveIndexMaterial:
    """Wrapper around the `refractiveindex` database material interface.

    Parameters
    ----------
    shelf : str
        Refractiveindex.info shelf name.
    book : str
        Refractiveindex.info book name.
    page : str
        Refractiveindex.info page name.
    **ri_kwargs
        Additional keyword arguments forwarded to
        :class:`refractiveindex.refractiveindex.RefractiveIndex`.
    """
    def __init__(self, shelf, book, page, **ri_kwargs):
        if not _HAS_REFRACTIVEINDEX:
            raise ImportError(
                "RefractiveIndexMaterial requires the 'refractiveindex' package. "
                "Install it with: pip install refractiveindex"
            )
        self.material = RIMaterial(shelf=shelf, book=book, page=page, **ri_kwargs)
        self.rangeMin, self.rangeMax = self.material._wl_range
        
        # Guard against NaNs at the edges of the wavelength range by slightly shrinking it if needed
        if np.isnan(self.get_eps(self.rangeMin*1e-6)):
            self.rangeMin += 0.1
        if np.isnan(self.get_eps(self.rangeMax*1e-6)):
            self.rangeMax -= 0.1

    def get_refractive_index(self, wl):
        """Return refractive index values from the database.

        Parameters
        ----------
        wl : float | array-like
            Wavelength in nanometers.

        Returns
        -------
        float | numpy.ndarray
            Refractive index ``n``.
        """
        return self.material.get_refractive_index(np.copy(wl))
    
    def get_extinction_coefficient(self, wl):
        """Return extinction coefficient values from the database.

        Parameters
        ----------
        wl : float | array-like
            Wavelength in nanometers.

        Returns
        -------
        float | numpy.ndarray
            Extinction coefficient ``k``.
        """
        return self.material.get_extinction_coefficient(np.copy(wl))

    def get_eps(self, wavelength, exp_type='exp_minus_i_omega_t', real=True):
        """Evaluate permittivity from database material data.

        Parameters
        ----------
        wavelength : float | array-like
            Wavelength in meters (or astropy quantity convertible to meters).
        exp_type : {"exp_minus_i_omega_t", "exp_plus_i_omega_t"}, default="exp_minus_i_omega_t"
            Time-harmonic convention used for complex permittivity sign.
        real : bool, default=True
            If ``True``, returns ``n^2``. If ``False``, returns ``(n ± i k)^2``.

        Returns
        -------
        float | complex | numpy.ndarray
            Relative permittivity.
        """
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

    @classmethod
    def from_upstream(cls, material):
        """Build wrapper from an upstream ``refractiveindex`` material instance."""
        obj = cls.__new__(cls)
        obj.material = material
        wl_range = getattr(material, "_wl_range", (None, None))
        if wl_range is None:
            wl_range = (None, None)
        obj.rangeMin, obj.rangeMax = wl_range
        return obj


class StepIndexFibre:
    """Cylindrical step-index fibre model with analytical mode utilities.

    The class stores core/cladding constitutive parameters and provides methods
    to compute dispersion roots, propagation constants, and mode objects.
    """
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
        core : RefractiveIndexMaterial, optional
            Core material object (highest precedence for the core side).
        clad : RefractiveIndexMaterial, optional
            Cladding material object (highest precedence for the cladding side).
        eps_core : float or callable, optional
            Core relative permittivity ``eps(lambda)`` if ``core`` is not provided.
        eps_clad : float or callable, optional
            Cladding relative permittivity ``eps(lambda)`` if ``clad`` is not provided.
        n_core : float or callable, optional
            Core refractive index ``n(lambda)`` used if neither ``core`` nor
            ``eps_core`` is provided.
        n_clad : float or callable, optional
            Cladding refractive index ``n(lambda)`` used if neither ``clad`` nor
            ``eps_clad`` is provided.
        mu_core : float or callable, default=1.0
            Core relative permeability ``mu(lambda)``.
        mu_clad : float or callable, default=1.0
            Cladding relative permeability ``mu(lambda)``.
        exp_type : str, default="exp_minus_i_omega_t"
            Passed to ``RefractiveIndexMaterial.get_eps`` when ``core``/``clad`` are provided.
        real_eps_from_material : if True, use n^2 (real). If False and k is available in
                                 the database, use (n ± i k)^2 depending on exp_type.
        """

        self.core_radius = _strip_unit(core_radius, units.m if _HAS_UNITS else None)
        
        def _normalize_material(mat, side):
            if mat is None:
                return None
            if isinstance(mat, RefractiveIndexMaterial):
                return mat
            if _HAS_REFRACTIVEINDEX and isinstance(mat, RIMaterial):
                warnings.warn(
                    f"{side}: received upstream RefractiveIndexMaterial (nm API); "
                    "auto-wrapping to anafibre RefractiveIndexMaterial (m API).",
                    UserWarning,
                    stacklevel=2,
                )
                return RefractiveIndexMaterial.from_upstream(mat)
            raise TypeError(
                f"{side} must be anafibre.RefractiveIndexMaterial or None; "
                f"got {type(mat).__name__}"
            )

        core = _normalize_material(core, "core")
        clad = _normalize_material(clad, "clad")
        self.core = core
        self.clad = clad



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
        # Store μ (can be scalar or callable)
        self.mu_core = _as_callable(mu_core)
        self.mu_clad = _as_callable(mu_clad)
        self.eps_core = _pick_eps("core", core, eps_core, n_core, self.mu_core)
        self.eps_clad = _pick_eps("clad", clad, eps_clad, n_clad, self.mu_clad)

    def _eval(self, val, wavelength):
        if callable(val):
            return val(wavelength)
        return val

    def n_core(self, wavelength):
        """Return core refractive index at a wavelength.

        Parameters
        ----------
        wavelength : float | array-like
            Wavelength in meters.

        Returns
        -------
        float | complex | numpy.ndarray
            Core refractive index.
        """
        wl = _strip_unit(wavelength, units.m if _HAS_UNITS else None)
        eps = self._eval(self.eps_core, wl)
        mu = self._eval(self.mu_core, wl)
        return np.sqrt(eps * mu)

    def n_clad(self, wavelength):
        """Return cladding refractive index at a wavelength.

        Parameters
        ----------
        wavelength : float | array-like
            Wavelength in meters.

        Returns
        -------
        float | complex | numpy.ndarray
            Cladding refractive index.
        """
        wl = _strip_unit(wavelength, units.m if _HAS_UNITS else None)
        eps = self._eval(self.eps_clad, wl)
        mu = self._eval(self.mu_clad, wl)
        return np.sqrt(eps * mu)

    def eps(self, r, wavelength):
        """Return radial relative permittivity profile.

        Parameters
        ----------
        r : float | array-like
            Radial coordinate in meters.
        wavelength : float | array-like
            Wavelength in meters.

        Returns
        -------
        float | complex | numpy.ndarray
            Piecewise ``eps_core`` inside the core and ``eps_clad`` outside.
        """
        rr = _strip_unit(r, units.m if _HAS_UNITS else None)
        wl = _strip_unit(wavelength, units.m if _HAS_UNITS else None)
        return np.where(rr < self.core_radius,
                        self._eval(self.eps_core, wl),
                        self._eval(self.eps_clad, wl))

    def mu(self, r, wavelength):
        """Return radial relative permeability profile.

        Parameters
        ----------
        r : float | array-like
            Radial coordinate in meters.
        wavelength : float | array-like
            Wavelength in meters.

        Returns
        -------
        float | complex | numpy.ndarray
            Piecewise ``mu_core`` inside the core and ``mu_clad`` outside.
        """
        rr = _strip_unit(r, units.m if _HAS_UNITS else None)
        wl = _strip_unit(wavelength, units.m if _HAS_UNITS else None)
        return np.where(rr < self.core_radius,
                        self._eval(self.mu_core, wl),
                        self._eval(self.mu_clad, wl))

    def n(self, r, wavelength):
        """Return radial refractive index profile.

        Parameters
        ----------
        r : float | array-like
            Radial coordinate in meters.
        wavelength : float | array-like
            Wavelength in meters.

        Returns
        -------
        float | complex | numpy.ndarray
            Refractive index profile ``sqrt(eps * mu)``.
        """
        return np.sqrt(self.eps(r, wavelength) * self.mu(r, wavelength))

    def V(self, wavelength):
        """Compute normalized frequency parameter ``V``.

        Parameters
        ----------
        wavelength : float | array-like
            Wavelength in meters.

        Returns
        -------
        float | numpy.ndarray
            Normalized frequency ``V = k0 a sqrt(n_core^2 - n_clad^2)``.

        Notes
        -----
        This corresponds to the paper definition
        $V=k_0\\rho_0\\sqrt{n_1^2-n_2^2}$,
        where $\\rho_0$ is the core radius.
        """
        wl = _strip_unit(wavelength, units.m if _HAS_UNITS else None)
        n1 = self.n_core(wl)
        n2 = self.n_clad(wl)
        k0 = 2 * np.pi / wl
        return k0 * self.core_radius * np.sqrt(n1**2 - n2**2)

    def wavelength_from_V_legacy(self, V):
        """Closed-form inversion of ``V`` for constant-index fibres.

        Parameters
        ----------
        V : float | array-like
            Normalized frequency.

        Returns
        -------
        float | numpy.ndarray
            Wavelength in meters.

        Raises
        ------
        ValueError
            If ``n_core <= n_clad``.
        """
        # raise ValueError("Ha! This method is not implemented in StepIndexFibre. Use a different method to compute wavelength from V.")
        n1 = self.n_core(1.0)
        n2 = self.n_clad(1.0)
        a = self.core_radius
        delta_n2 = n1**2 - n2**2
        if delta_n2 <= 0:
            raise ValueError("Core index must be larger than cladding index.")
        V = np.asarray(V)
        result = np.empty_like(V, dtype=float)
        result[V == 0] = np.inf
        result[V != 0] = 2 * np.pi * a * np.sqrt(delta_n2) / V[V != 0]
        return result if result.shape != () else result.item()

    def wavelength_from_V(
        self,
        V,
        wl_bracket=(2e-7, 5e-6),  # meters: search range if indices are callable
        rtol=1e-12,
        maxiter=100,
        _grid_pts=64,
    ):
        """
        Invert ``V(lambda)`` to obtain wavelength.

        Parameters
        ----------
        V : float | array-like
            Target normalized frequency value(s).
        wl_bracket : tuple[float, float], default=(2e-7, 5e-6)
            Search interval in meters used when material properties are wavelength-dependent.
        rtol : float, default=1e-12
            Relative tolerance for scalar root solving.
        maxiter : int, default=100
            Maximum iterations for scalar root solving.
        _grid_pts : int, default=64
            Number of logarithmic sampling points used to bracket roots.

        Returns
        -------
        float | numpy.ndarray
            Wavelength in meters. ``V == 0`` maps to ``np.inf``.

        Raises
        ------
        ValueError
            If the bracket is invalid or no wavelength in the bracket reaches a target ``V``.

        Notes
        -----
        Solves the implicit equation
        $V(\\lambda)=\\frac{2\\pi a}{\\lambda}\\sqrt{n_\\mathrm{core}(\\lambda)^2-n_\\mathrm{clad}(\\lambda)^2}$.
        For constant material parameters, the closed-form inverse is used.
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

    def b(self, ell, m, V=None, wavelength=None, mode_type=None,
          N_b=2000, tol=1e-15, complex_tol=1e-8, maxiter=100):
        """Return normalized propagation constant ``b`` for a guided mode.

        Parameters
        ----------
        ell : int
            Azimuthal mode index.
        m : int
            Radial mode index (1-based).
        V : float | array-like, optional
            Normalized frequency values.
        wavelength : float | array-like, optional
            Wavelength values in meters.
        mode_type : {"TE", "TM", None}, optional
            Mode family selector for ``ell == 0``.
        N_b : int, default=2000
            Number of samples used for root bracketing.
        tol : float, default=1e-15
            Real root-finder tolerance.
        complex_tol : float, default=1e-8
            Complex root acceptance tolerance.
        maxiter : int, default=100
            Maximum iterations for complex root refinement.

        Returns
        -------
        float | complex | numpy.ndarray
            Guided root(s) ``b``.

        Notes
        -----
        $b$ is the normalized propagation constant used in the paper:
        $b=\\frac{n_\\mathrm{eff}^2-n_2^2}{n_1^2-n_2^2}$, constrained to $[0,1]$ for
        guided modes in lossless conditions.
        """
        if V is not None:
            return find_b_of_V(self, ell, m, V=V, mode_type=mode_type, N_b=N_b, tol=tol, complex_tol=complex_tol, maxiter=maxiter)
        elif wavelength is not None:
            return find_b_of_V(self, ell, m, wavelength=wavelength, mode_type=mode_type, N_b=N_b, tol=tol, complex_tol=complex_tol, maxiter=maxiter)
        else:
            raise ValueError("Specify either V or wavelength.")

    def neff(self, ell, m, V=None, wavelength=None, mode_type=None, **kwargs):
        """Return effective index for a selected guided mode.

        Parameters
        ----------
        ell, m : int
            Azimuthal and radial mode indices.
        V : float | array-like, optional
            Normalized frequency values.
        wavelength : float | array-like, optional
            Wavelength values in meters.
        mode_type : {"TE", "TM", None}, optional
            Mode family selector for ``ell == 0``.
        **kwargs
            Forwarded to :meth:`b` (for example ``N_b``, ``tol``).

        Returns
        -------
        float | complex | numpy.ndarray
            Effective index ``n_eff``.

        Notes
        -----
        Equivalent to composing :meth:`b` with
        $n_\\mathrm{eff}(b)=\\sqrt{b\\,n_1^2+(1-b)\\,n_2^2}$.
        """
        if wavelength is not None:
            wl = wavelength
        elif V is not None:
            wl = self.wavelength_from_V(V)
        else:
            raise ValueError("Specify either V or wavelength.")
        b_val = self.b(ell, m, V=V, wavelength=wavelength, mode_type=mode_type, **kwargs)
        return b_to_neff(self, b_val, wl)

    def kz(self, ell, m, V=None, wavelength=None, mode_type=None, **kwargs):
        """Return longitudinal propagation constant for a guided mode.

        Parameters
        ----------
        ell, m : int
            Azimuthal and radial mode indices.
        V : float | array-like, optional
            Normalized frequency values.
        wavelength : float | array-like, optional
            Wavelength values in meters.
        mode_type : {"TE", "TM", None}, optional
            Mode family selector for ``ell == 0``.
        **kwargs
            Forwarded to :meth:`b` (for example ``N_b``, ``tol``).

        Returns
        -------
        float | complex | numpy.ndarray
            Propagation constant ``k_z`` in rad/m.

        Notes
        -----
        Uses $k_z=n_\\mathrm{eff}k_0$ with $k_0=2\\pi/\\lambda$.
        """
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

        Notes
        -----
        This is the API-level wrapper of :func:`anafibre.dispersion.F_dispersion`,
        corresponding to the regularized scalar dispersion function used for robust
        root-finding in the paper.
        """
        return F_dispersion(self, ell, b, V=V, wavelength=wavelength, mode_type=mode_type)

    def nu_vs_V(self, ell, m, V_vals, *, mode_type=None):
        """
        Compute the hybrid-mode parameter ``nu`` as a function of $V$.

        Parameters
        ----------
        ell : int
            Azimuthal mode index (must be non-zero).
        m : int
            Radial mode index.
        V_vals : array-like
            Normalized frequency samples.
        mode_type : {"TE", "TM", None}, optional
            Reserved for consistency; used when resolving ``b(V)``.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(nu, mask_valid)`` where ``mask_valid`` marks finite guided solutions.

        Notes
        -----
        For hybrid modes, this follows the analytical amplitude ratio used in the
        paper:

        - $\\nu_\\varepsilon=i\\,\\phi_\\varepsilon/(\\ell n_\\mathrm{eff})$
        - $\\nu=\\sqrt{-\\phi_\\varepsilon/\\phi_\\mu+0j}$

        with branch sign fixed by $\\mathrm{Re}(\\nu\\,\\nu_\\varepsilon^*)\\ge 0$.
        """
        import numpy as np
        from scipy.special import jv, jvp, kve

        if int(ell) == 0:
            raise ValueError("nu is only defined for ell != 0 (hybrid modes).")

        V = np.asarray(V_vals, dtype=float)

        # Map V -> wavelength and b for this (ell, m)
        wl = self.wavelength_from_V_legacy(V)
        b = self.b(ell, m, V=V, mode_type=mode_type)

        # Valid guided region
        mask_b = np.isfinite(b) & (b > 0) & (b < 1)

        # Effective index
        ne = b_to_neff(self, b, wl)

        # Material parameters at wl
        eps1 = self._eval(self.eps_core, wl)
        eps2 = self._eval(self.eps_clad, wl)
        mu1  = self._eval(self.mu_core, wl)
        mu2  = self._eval(self.mu_clad, wl)

        # u, w and derivatives
        u = V * np.sqrt(1 - b)
        w = V * np.sqrt(b)

        with np.errstate(all='ignore'):
            DlnJ = jvp(ell, u) / jv(ell, u)
            DlnK = -(kve(ell - 1, w) + kve(ell + 1, w)) / (2 * kve(ell, w))

            fac = (u * w / V) ** 2
            phi_eps = fac * (eps1 / u * DlnJ + eps2 / w * DlnK)
            phi_mu  = fac * (mu1  / u * DlnJ + mu2  / w * DlnK)

            nu_eps = 1j * phi_eps / (ell * ne)
            nu = np.sqrt(-phi_eps / phi_mu + 0j)
            s = np.real(nu * np.conj(nu_eps))
            flip = s < 0
            nu = np.where(flip, -nu, nu)

        mask_finite = np.isfinite(wl) & np.isfinite(ne) & np.isfinite(phi_eps) & np.isfinite(phi_mu)
        mask_valid = mask_b & mask_finite
        return nu, mask_valid

    def sigma_vs_V(self, ell, m, V_vals, *, mode_type=None, atol=1e-9):
        """
        Compute mode normalization coefficient $\\sigma(V)$.

        Parameters
        ----------
        ell : int
            Azimuthal mode index.
        m : int
            Radial mode index.
        V_vals : array-like
            Normalized frequency samples.
        mode_type : {"TE", "TM", None}, optional
            Required for robust ``ell == 0`` classification.
        atol : float, default=1e-9
            Tolerance used when inferring TE/TM from dispersion terms.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(sigma, mask_valid)`` where ``mask_valid`` marks finite guided solutions.

        Notes
        -----
        $\\sigma$ is the modal normalization coefficient entering
        $P=c_0|N|^2\\sigma$ in the paper's power normalization derivation.
        The implementation uses the closed-form $I_+$ and $I_-$ radial-integral
        expressions (no numerical quadrature).
        """
        import numpy as np
        from scipy.special import jv, jvp, kve
        from scipy.constants import c as c0
        from .dispersion import _wDlnK

        V = np.asarray(V_vals, dtype=float)
        wl = self.wavelength_from_V_legacy(V)
        b = self.b(ell, m, V=V, mode_type=mode_type)

        # Guided region mask
        mask_b = np.isfinite(b) & (b > 0) & (b < 1)

        # Indices and wavenumbers
        n1 = self.n_core(wl)
        n2 = self.n_clad(wl)
        k0 = 2*np.pi / wl
        k1 = k0 * n1
        k2 = k0 * n2

        # kz and neff
        ne = b_to_neff(self, b, wl)
        kz = b_to_kz(self, b, wl)

        # Materials
        eps1 = self._eval(self.eps_core, wl)
        eps2 = self._eval(self.eps_clad, wl)
        mu1  = self._eval(self.mu_core, wl)
        mu2  = self._eval(self.mu_clad, wl)

        # u, w and derivatives
        with np.errstate(all='ignore'):
            u = V * np.sqrt(1 - b)
            w = V * np.sqrt(b)
            DlnJ = jvp(ell, u) / jv(ell, u)
            # DlnK = -(kve(ell - 1, w) + kve(ell + 1, w)) / (2 * kve(ell, w))
            DlnK = _wDlnK(ell, w)/w
            fac = (u * w / V) ** 2
            phi_eps = fac * (eps1 / u * DlnJ + eps2 / w * DlnK)
            phi_mu  = fac * (mu1  / u * DlnJ + mu2  / w * DlnK)

        # Amplitudes A,B (unnormalised) per fields._compute_amplitudes
        A = np.zeros_like(V, dtype=complex)
        B = np.zeros_like(V, dtype=complex)
        if ell == 0:
            mt = (mode_type or "").lower() if mode_type is not None else None
            if mt is None:
                # Try heuristic inference per element
                is_te = np.isclose(phi_mu, 0, atol=atol)
                is_tm = np.isclose(phi_eps, 0, atol=atol)
                # Default to NaN where ambiguous
                A[is_tm & ~is_te] = 1.0
                B[is_tm & ~is_te] = 0.0
                A[is_te & ~is_tm] = 0.0
                B[is_te & ~is_tm] = 1.0j
                # Leave others as NaN to be masked later
                A[~(is_te ^ is_tm)] = np.nan
                B[~(is_te ^ is_tm)] = np.nan
            elif mt == 'te':
                A[:] = 0.0
                B[:] = 1.0j
            elif mt == 'tm':
                A[:] = 1.0
                B[:] = 0.0
            else:
                raise ValueError("mode_type must be 'TE' or 'TM' for ell = 0")
        else:
            with np.errstate(all='ignore'):
                nu_eps = 1j * phi_eps / (ell * ne)
                nu = np.sqrt(-phi_eps / phi_mu + 0j)
                s = np.real(nu * np.conj(nu_eps))
                nu = np.where(s < 0, -nu, nu)
                A = 1.0 / np.sqrt(1.0 + np.abs(nu)**2)
                B = nu * A

        # Integrals and coefficients
        with np.errstate(all='ignore', divide='ignore', invalid='ignore'):
            I1_plus  = (u**2 - ell**2) / (2 * u**2) + DlnJ / u + (DlnJ**2) / 2
            I1_minus = ell / (u**2)
            I2_plus  = (w**2 + ell**2) / (2 * w**2) - DlnK / w - (DlnK**2) / 2
            I2_minus = -ell / (w**2)

            denAB = (np.abs(A)**2 + np.abs(B)**2)
            alpha1_plus = (eps1 * np.abs(A)**2 + mu1 * np.abs(B)**2) / denAB
            alpha2_plus = (eps2 * np.abs(A)**2 + mu2 * np.abs(B)**2) / denAB
            alpha_minus  = (np.imag(A * np.conj(B))) / denAB

            kappasq1 = k1**2 - kz**2
            gammasq2 = kz**2 - k2**2

            term1 = (kz * k0) / kappasq1 * alpha1_plus * I1_plus
            term2 = (kz**2 + k1**2) / kappasq1 * alpha_minus * I1_minus
            term3 = (kz * k0) / gammasq2 * alpha2_plus * I2_plus
            term4 = (kz**2 + k2**2) / gammasq2 * alpha_minus * I2_minus

            sigma = c0 * np.pi * (self.core_radius**2) * (term1 + term2 + term3 + term4)

        # Validity mask
        finite_parts = (
            np.isfinite(wl) & np.isfinite(ne) & np.isfinite(phi_eps) & np.isfinite(phi_mu)
            & np.isfinite(kz) & np.isfinite(k1) & np.isfinite(k2)
            & np.isfinite(sigma)
        )
        mask_valid = mask_b & finite_parts
        return np.real(sigma), mask_valid
    
    def m_max(self, ell, wavelength, mode_type=None, N_b=2000, tol=1e-15, complex_tol=1e-8, maxiter=100):
        """
        Find the highest guided radial index ``m`` for fixed ``ell`` and wavelength.

        Parameters
        ----------
        ell : int
            Azimuthal mode index.
        wavelength : float
            Wavelength in meters.
        mode_type : {"TE", "TM", None}, optional
            Mode family selector for ``ell == 0``.
        N_b, tol, complex_tol, maxiter
            Root-finding controls passed to :meth:`b`.

        Returns
        -------
        int
            Largest guided radial index, or ``0`` if no guided solution exists.
        """
        def _is_guided_root(b_val):
            if not np.isfinite(b_val):
                return False
            b_re = float(np.real(b_val))
            b_im = float(np.imag(b_val))
            return (0 < b_re < 1) and (abs(b_im) <= complex_tol)

        m = 1
        while True:
            b = self.b(ell, m, wavelength=wavelength, mode_type=mode_type, N_b=N_b, tol=tol, complex_tol=complex_tol, maxiter=maxiter)
            if not _is_guided_root(b):
                break
            m += 1
        return m - 1  # The last valid one
    
    def ell_max(self, wavelength, m=1, mode_type=None, N_b=2000, tol=1e-15, complex_tol=1e-8, maxiter=100, ell_max_search=100):
        """
        Find the highest guided azimuthal index ``ell`` for fixed radial order.

        Parameters
        ----------
        wavelength : float
            Wavelength in meters.
        m : int, default=1
            Radial mode index to track while increasing ``ell``.
        mode_type : {"TE", "TM", None}, optional
            Mode family selector for ``ell == 0``.
        N_b, tol, complex_tol, maxiter
            Root-finding controls passed to :meth:`b`.
        ell_max_search : int, default=100
            Maximum ``ell`` value checked before stopping.

        Returns
        -------
        int
            Largest guided azimuthal index found, or ``-1`` if none exists.
        """
        def _is_guided_root(b_val):
            if not np.isfinite(b_val):
                return False
            b_re = float(np.real(b_val))
            b_im = float(np.imag(b_val))
            return (0 < b_re < 1) and (abs(b_im) <= complex_tol)

        def _has_mode(ell):
            if ell == 0 and mode_type is None:
                # For ell=0, TE/TM may not both exist; accept either.
                b_te = self.b(0, m, wavelength=wavelength, mode_type="TE", N_b=N_b, tol=tol, complex_tol=complex_tol, maxiter=maxiter)
                b_tm = self.b(0, m, wavelength=wavelength, mode_type="TM", N_b=N_b, tol=tol, complex_tol=complex_tol, maxiter=maxiter)
                return _is_guided_root(b_te) or _is_guided_root(b_tm)
            b = self.b(ell, m, wavelength=wavelength, mode_type=mode_type, N_b=N_b, tol=tol, complex_tol=complex_tol, maxiter=maxiter)
            return _is_guided_root(b)

        max_valid_ell = -1
        found_positive_ell = False

        for ell in range(ell_max_search):
            has_mode = _has_mode(ell)
            if has_mode:
                max_valid_ell = ell
                if ell > 0:
                    found_positive_ell = True
            elif ell > 0 and found_positive_ell:
                # For fixed wavelength and m, guided existence is monotonic in ell.
                break

        return max_valid_ell
    
    def list_modes_at(self, wavelength):
        """Enumerate all guided modes at a given wavelength.

        Parameters
        ----------
        wavelength : float
            Wavelength in meters.

        Returns
        -------
        anafibre.utils.GuidedModeList
            Guided modes sorted by decreasing ``n_eff``.
        """
        from .utils import GuidedModeList

        ## Calculate and display guided modes
        modes = []       # List to store the modes
        ell_max = self.ell_max(wavelength=wavelength)  # Maximum azimuthal mode number
        for ell in range(0, ell_max + 1):       # Loop over azimuthal mode numbers
            if ell == 0:    # Special case for ell = 0
                m_max_te = self.m_max(wavelength=wavelength, ell=ell, mode_type="TE")  # Maximum radial mode number for TE
                m_max_tm = self.m_max(wavelength=wavelength, ell=ell, mode_type="TM")  # Maximum radial mode number for TM
                for m in range(1, m_max_te + 1):    
                    modes.append(self.TE(n=m, wl=wavelength))  
                for m in range(1, m_max_tm + 1):
                    modes.append(self.TM(n=m, wl=wavelength))
            else:   
                m_max = self.m_max(wavelength=wavelength, ell=ell) # Maximum radial mode number for hybrid modes
                for m in range(1, m_max + 1):
                    if m % 2 == 1:  # Odd m corresponds to HE modes
                        modes.append(self.HE(ell=ell, n=m, wl=wavelength)) 
                    else:           # Even m corresponds to EH modes
                        modes.append(self.EH(ell=ell, n=m, wl=wavelength))

        modes = [m for m in modes if m is not None]                 # Remove any None entries from the modes list
        modes = sorted(modes, key=lambda m: m.neff, reverse=True)   # Sort modes by effective index in descending order
        return GuidedModeList(modes)
    # -------------------------- mode construction ----------------------------
    def HE(self, ell, n, wl, **kwargs):
        """Construct an ``HE_{ell,n}`` mode.

        Parameters
        ----------
        ell : int
            Azimuthal mode index (``ell > 0``).
        n : int
            Radial index in HE/EH notation (mapped to internal odd ``m``).
        wl : float
            Wavelength in meters.
        **kwargs
            Forwarded to :class:`anafibre.fields.GuidedMode`.

        Returns
        -------
        anafibre.fields.GuidedMode
            Guided HE mode object.
        """
        m = 2 * n - 1
        return GuidedMode(self, ell=ell, m=m, wavelength=wl,
                          mode_type="HE", **kwargs)

    def EH(self, ell, n, wl, **kwargs):
        """Construct an ``EH_{ell,n}`` mode.

        Parameters
        ----------
        ell : int
            Azimuthal mode index (``ell > 0``).
        n : int
            Radial index in HE/EH notation (mapped to internal even ``m``).
        wl : float
            Wavelength in meters.
        **kwargs
            Forwarded to :class:`anafibre.fields.GuidedMode`.

        Returns
        -------
        anafibre.fields.GuidedMode
            Guided EH mode object.
        """
        m = 2 * n
        return GuidedMode(self, ell=ell, m=m, wavelength=wl,
                          mode_type="EH", **kwargs)

    def TE(self, n, wl, **kwargs):
        """Construct a ``TE_{0,n}`` mode.

        Parameters
        ----------
        n : int
            Radial mode index.
        wl : float
            Wavelength in meters.
        **kwargs
            Forwarded to :class:`anafibre.fields.GuidedMode`.

        Returns
        -------
        anafibre.fields.GuidedMode
            Guided TE mode object.
        """
        return GuidedMode(self, ell=0, m=n, wavelength=wl,
                          mode_type="TE", **kwargs)

    def TM(self, n, wl, **kwargs):
        """Construct a ``TM_{0,n}`` mode.

        Parameters
        ----------
        n : int
            Radial mode index.
        wl : float
            Wavelength in meters.
        **kwargs
            Forwarded to :class:`anafibre.fields.GuidedMode`.

        Returns
        -------
        anafibre.fields.GuidedMode
            Guided TM mode object.
        """
        return GuidedMode(self, ell=0, m=n, wavelength=wl,
                          mode_type="TM", **kwargs)
    
    def __repr__(self):
        return (f"StepIndexFibre(core_radius={self.core_radius:.3e}, "
                f"eps_core={self.eps_core}, eps_clad={self.eps_clad}, "
                f"mu_core={self.mu_core}, mu_clad={self.mu_clad})")
