"""
Defines the StepIndexFibre class for cylindrical step-index optical fibers.
This class includes methods to compute refractive indices, normalised frequency $V$,
normalised propagation constant $b$, effective index $n_\\mathrm{e}$, propagation constant $k_z$,
and electromagnetic fields.
"""

import numpy as np
import warnings
from .dispersion import find_b_of_V, b_to_neff, b_to_kz, F_dispersion, Phi_alpha
from .fields import GuidedMode
from .utils import units, _HAS_UNITS, _strip_unit, _HAS_REFRACTIVEINDEX, RIMaterial, NoExtinctionCoefficient
from scipy.optimize import brentq
  
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
        self._validate_upstream_api(self.material)
        self._set_range_from_upstream()

    @staticmethod
    def _wl_to_m(wl):
        """Convert wavelength input to meters for upstream API calls."""
        return np.copy(_strip_unit(wl, units.m if _HAS_UNITS else None))

    @staticmethod
    def _validate_upstream_api(material):
        required = (
            "get_refractive_index",
            "get_extinction_coefficient",
            "get_epsilon",
            "get_wl_range",
        )
        missing = [name for name in required if not hasattr(material, name)]
        if missing:
            raise ImportError(
                "Detected an incompatible 'refractiveindex' API. "
                "This version of anafibre requires refractiveindex >= 1.0.2 "
                f"(missing: {', '.join(missing)})."
            )

    @staticmethod
    def _range_m_from_upstream(material):
        wl_range = material.get_wl_range(unit="m")
        if wl_range is None:
            return None, None
        wl_min, wl_max = wl_range
        if wl_min is None or wl_max is None:
            return None, None
        wl_min = float(wl_min)
        wl_max = float(wl_max)
        return (wl_min, wl_max) if wl_min <= wl_max else (wl_max, wl_min)

    def _set_range_from_upstream(self):
        self.rangeMin, self.rangeMax = self._range_m_from_upstream(self.material)
        if self.rangeMin is None or self.rangeMax is None:
            return

        # Guard against NaNs at range edges by slightly shrinking the interval.
        edge_pad_m = 0.1e-6
        if np.isnan(self.get_eps(self.rangeMin)):
            self.rangeMin += edge_pad_m
        if np.isnan(self.get_eps(self.rangeMax)):
            self.rangeMax -= edge_pad_m

    def get_refractive_index(self, wl):
        """Return refractive index values from the database.

        Parameters
        ----------
        wl : float | array-like
            Wavelength in meters (or astropy quantity convertible to meters).

        Returns
        -------
        float | numpy.ndarray
            Refractive index ``n``.
        """
        return self.material.get_refractive_index(self._wl_to_m(wl), unit="m")
    
    def get_extinction_coefficient(self, wl):
        """Return extinction coefficient values from the database.

        Parameters
        ----------
        wl : float | array-like
            Wavelength in meters (or astropy quantity convertible to meters).

        Returns
        -------
        float | numpy.ndarray
            Extinction coefficient ``k``.
        """
        return self.material.get_extinction_coefficient(self._wl_to_m(wl), unit="m")

    def get_eps(self, wl, exp_type='exp_minus_i_omega_t', real=True):
        """Evaluate permittivity from database material data.

        Parameters
        ----------
        wl : float | array-like
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
        if real:
            n = self.get_refractive_index(wl)
            return n**2
        try:
            return self.material.get_epsilon(self._wl_to_m(wl), unit="m", exp_type=exp_type)
        except NoExtinctionCoefficient:
            n = self.get_refractive_index(wl)
            return n**2

    @classmethod
    def from_upstream(cls, material):
        """Build wrapper from an upstream ``refractiveindex`` material instance."""
        obj = cls.__new__(cls)
        cls._validate_upstream_api(material)
        obj.material = material
        obj._set_range_from_upstream()
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

    def _eval(self, val, wl):
        if callable(val):
            return val(wl)
        return val

    def n_core(self, wl):
        """Return core refractive index at a wavelength.

        Parameters
        ----------
        wl : float | array-like
            Wavelength in meters.

        Returns
        -------
        float | complex | numpy.ndarray
            Core refractive index.
        """
        wl = _strip_unit(wl, units.m if _HAS_UNITS else None)
        eps = self._eval(self.eps_core, wl)
        mu = self._eval(self.mu_core, wl)
        return np.sqrt(eps * mu)

    def n_clad(self, wl):
        """Return cladding refractive index at a wavelength.

        Parameters
        ----------
        wl : float | array-like
            Wavelength in meters.

        Returns
        -------
        float | complex | numpy.ndarray
            Cladding refractive index.
        """
        wl = _strip_unit(wl, units.m if _HAS_UNITS else None)
        eps = self._eval(self.eps_clad, wl)
        mu = self._eval(self.mu_clad, wl)
        return np.sqrt(eps * mu)

    def eps(self, r, wl):
        """Return radial relative permittivity profile.

        Parameters
        ----------
        r : float | array-like
            Radial coordinate in meters.
        wl : float | array-like
            Wavelength in meters.

        Returns
        -------
        float | complex | numpy.ndarray
            Piecewise ``eps_core`` inside the core and ``eps_clad`` outside.
        """
        rr = _strip_unit(r, units.m if _HAS_UNITS else None)
        wl = _strip_unit(wl, units.m if _HAS_UNITS else None)
        return np.where(rr < self.core_radius,
                        self._eval(self.eps_core, wl),
                        self._eval(self.eps_clad, wl))

    def mu(self, r, wl):
        """Return radial relative permeability profile.

        Parameters
        ----------
        r : float | array-like
            Radial coordinate in meters.
        wl : float | array-like
            Wavelength in meters.

        Returns
        -------
        float | complex | numpy.ndarray
            Piecewise ``mu_core`` inside the core and ``mu_clad`` outside.
        """
        rr = _strip_unit(r, units.m if _HAS_UNITS else None)
        wl = _strip_unit(wl, units.m if _HAS_UNITS else None)
        return np.where(rr < self.core_radius,
                        self._eval(self.mu_core, wl),
                        self._eval(self.mu_clad, wl))

    def n(self, r, wl):
        """Return radial refractive index profile.

        Parameters
        ----------
        r : float | array-like
            Radial coordinate in meters.
        wl : float | array-like
            Wavelength in meters.

        Returns
        -------
        float | complex | numpy.ndarray
            Refractive index profile ``sqrt(eps * mu)``.
        """
        return np.sqrt(self.eps(r, wl) * self.mu(r, wl))

    def V(self, wl):
        """Compute normalised frequency parameter ``V``.

        Parameters
        ----------
        wl : float | array-like
            Wavelength in meters.

        Returns
        -------
        float | numpy.ndarray
            normalised frequency ``V = k0 a sqrt(n_core^2 - n_clad^2)``.

        Notes
        -----
        This corresponds to the paper definition
        $V=k_0\\rho_0\\sqrt{n_1^2-n_2^2}$,
        where $\\rho_0$ is the core radius.
        """
        wl = _strip_unit(wl, units.m if _HAS_UNITS else None)
        n1 = self.n_core(wl)
        n2 = self.n_clad(wl)
        wl_arr = np.asarray(wl)
        valid_wl = np.isfinite(wl_arr) & (wl_arr > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            k0 = np.where(valid_wl, 2 * np.pi / wl_arr, np.nan)
        dn2 = np.real_if_close(n1**2 - n2**2)
        if np.iscomplexobj(dn2):
            dn2 = np.real(dn2)
        valid = np.isfinite(dn2) & (dn2 >= 0)
        safe_dn2 = np.where(valid, dn2, np.nan)
        with np.errstate(invalid="ignore"):
            root = np.sqrt(safe_dn2)
        return k0 * self.core_radius * root

    def wl_of_V(self, V,
        wl_bracket=None,  # meters: auto-overlap for materials, fallback to legacy default
        rtol=1e-12,
        maxiter=100,
        _grid_pts=64,
    ):
        """
        Invert ``V(lambda)`` to obtain wavelength. This function does not work well for V<~0.5 when the indices are wavelength-dependent, so use with caution in that regime.

        Parameters
        ----------
        V : float | array-like
            Target normalised frequency value(s).
        wl_bracket : tuple[float, float] | None, default=None
            Search interval in meters used when material properties are wavelength-dependent.
            If ``None``, attempts to use the overlap of ``core``/``clad`` material
            validity ranges and falls back to ``(2e-7, 5e-6)``.
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
        if wl_bracket is None:
            wl_min = wl_max = None
            if self.core is not None and self.clad is not None:
                cmin = getattr(self.core, "rangeMin", None)
                cmax = getattr(self.core, "rangeMax", None)
                dmin = getattr(self.clad, "rangeMin", None)
                dmax = getattr(self.clad, "rangeMax", None)
                if None not in (cmin, cmax, dmin, dmax):
                    cmin, cmax = float(cmin), float(cmax)
                    dmin, dmax = float(dmin), float(dmax)
                    if np.isfinite([cmin, cmax, dmin, dmax]).all():
                        wl_min = max(cmin, dmin)
                        wl_max = min(cmax, dmax)
            if wl_min is None or wl_max is None:
                wl_min, wl_max = (2e-7, 5e-6)
        else:
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
                valid_V = np.isfinite(V_span)
                if not np.any(valid_V):
                    raise ValueError(
                        f"No valid V(λ) samples in wl_bracket=[{wl_min:.3g}, {wl_max:.3g}] m. "
                        "Check material validity ranges and index contrast."
                    )
                Vmin = np.min(V_span[valid_V])
                Vmax = np.max(V_span[valid_V])
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

    def b(self, ell, m, V=None, wl=None, mode_type=None,
          N_b=2000, tol=1e-15, complex_tol=1e-8, maxiter=100):
        """Return normalised propagation constant ``b`` for a guided mode.

        Parameters
        ----------
        ell : int
            Azimuthal mode index.
        m : int
            Radial mode index (1-based).
        V : float | array-like, optional
            normalised frequency values.
        wl : float | array-like, optional
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
        $b$ is the normalised propagation constant used in the paper:
        $b=\\frac{n_\\mathrm{eff}^2-n_2^2}{n_1^2-n_2^2}$, constrained to $[0,1]$ for
        guided modes in lossless conditions.
        """
        if V is not None:
            return find_b_of_V(self, ell, m, V=V, mode_type=mode_type, N_b=N_b, tol=tol, complex_tol=complex_tol, maxiter=maxiter)
        elif wl is not None:
            return find_b_of_V(self, ell, m, wl=wl, mode_type=mode_type, N_b=N_b, tol=tol, complex_tol=complex_tol, maxiter=maxiter)
        else:
            raise ValueError("Specify either V or wavelength.")

    def neff(self, ell, m, V=None, wl=None, mode_type=None, **kwargs):
        """Return effective index for a selected guided mode.

        Parameters
        ----------
        ell, m : int
            Azimuthal and radial mode indices.
        V : float | array-like, optional
            normalised frequency values.
        wl : float | array-like, optional
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
        if wl is not None:
            wl = wl
        elif V is not None:
            wl = self.wavelength_from_V(V)
        else:
            raise ValueError("Specify either V or wavelength.")
        b_val = self.b(ell, m, V=V, wl=wl, mode_type=mode_type, **kwargs)
        return b_to_neff(self, b_val, wl)

    def kz(self, ell, m, V=None, wl=None, mode_type=None, **kwargs):
        """Return longitudinal propagation constant for a guided mode.

        Parameters
        ----------
        ell, m : int
            Azimuthal and radial mode indices.
        V : float | array-like, optional
            normalised frequency values.
        wl : float | array-like, optional
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
        if wl is not None:
            wl = wl
        elif V is not None:
            wl = self.wavelength_from_V(V)
        else:
            raise ValueError("Specify either V or wavelength.")
        b_val = self.b(ell, m, V=V, wl=wl, mode_type=mode_type, **kwargs)
        return b_to_kz(self, b_val, wl)
    
    def F(self, ell, b, V=None, wl=None, mode_type=None):
        """
        Return the dispersion function F(b, V) or F(b, wavelength) for given mode.

        Parameters
        ----------
        ell : int
            Azimuthal mode number.
        b : float or array-like
            normalised propagation constant.
        V : float, optional
            normalised frequency.
        wl : float, optional
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
        return F_dispersion(self, ell, b, V=V, wl=wl, mode_type=mode_type)

    def Phi(self, ell, b, V=None, wl=None, alpha=None):
        """Return :math:`\\Phi_\\ell^\\alpha` for the fibre dispersion model.

        Parameters
        ----------
        ell : int
            Azimuthal mode index.
        b : float | complex | array-like
            normalised propagation constant candidate(s).
        V : float | array-like, optional
            normalised frequency. Provide either ``V`` or ``wl``.
        wl : float | array-like, optional
            Wavelength in meters. Used when ``V`` is not provided.
        alpha : sequence
            Length-2 sequence ``[alpha1, alpha2]`` for core/cladding parameters.
            Each entry may be a scalar, array, or callable ``f(wl)``.

        Returns
        -------
        float | complex | numpy.ndarray
            Value(s) of :math:`\\Phi_\\ell^\\alpha`.
        """
        if wl is None and V is None:
            raise ValueError("Specify either V or wavelength.")
        if wl is None:
            wl = self.wl_of_V(V)
            Vnum = np.asarray(V)
        elif V is None:
            wl = _strip_unit(wl, units.m if _HAS_UNITS else None)
            Vnum = np.asarray(self.V(wl))
        else:
            wl = _strip_unit(wl, units.m if _HAS_UNITS else None)
            Vnum = np.asarray(V)

        try:
            a1, a2 = alpha
        except Exception as exc:
            raise ValueError("alpha must be a length-2 sequence [alpha1, alpha2].") from exc
        if callable(a1):
            a1 = a1(wl)
        if callable(a2):
            a2 = a2(wl)

        return Phi_alpha(ell=ell, b=b, V=Vnum, alpha=[a1, a2])

    def m_max(self, ell, wl, mode_type=None, N_b=2000, tol=1e-15, complex_tol=1e-8, maxiter=100):
        """
        Find the highest guided radial index ``m`` for fixed ``ell`` and wavelength.

        Parameters
        ----------
        ell : int
            Azimuthal mode index.
        wl : float
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
            b = self.b(ell, m, wl=wl, mode_type=mode_type, N_b=N_b, tol=tol, complex_tol=complex_tol, maxiter=maxiter)
            if not _is_guided_root(b):
                break
            m += 1
        return m - 1  # The last valid one
    
    def ell_max(self, wl, m=1, mode_type=None, N_b=2000, tol=1e-15, complex_tol=1e-8, maxiter=100, ell_max_search=100):
        """
        Find the highest guided azimuthal index ``ell`` for fixed radial order.

        Parameters
        ----------
        wl : float
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
                b_te = self.b(0, m, wl=wl, mode_type="TE", N_b=N_b, tol=tol, complex_tol=complex_tol, maxiter=maxiter)
                b_tm = self.b(0, m, wl=wl, mode_type="TM", N_b=N_b, tol=tol, complex_tol=complex_tol, maxiter=maxiter)
                return _is_guided_root(b_te) or _is_guided_root(b_tm)
            b = self.b(ell, m, wl=wl, mode_type=mode_type, N_b=N_b, tol=tol, complex_tol=complex_tol, maxiter=maxiter)
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
    
    def list_modes_at(self, wl):
        """Enumerate all guided modes at a given wavelength.

        Parameters
        ----------
        wl : float
            Wavelength in meters.

        Returns
        -------
        anafibre.utils.GuidedModeList
            Guided modes sorted by decreasing ``n_eff``.
        """
        from .utils import GuidedModeList

        ## Calculate and display guided modes
        modes = []       # List to store the modes
        ell_max = self.ell_max(wl=wl)  # Maximum azimuthal mode number
        for ell in range(0, ell_max + 1):       # Loop over azimuthal mode numbers
            if ell == 0:    # Special case for ell = 0
                m_max_te = self.m_max(wl=wl, ell=ell, mode_type="TE")  # Maximum radial mode number for TE
                m_max_tm = self.m_max(wl=wl, ell=ell, mode_type="TM")  # Maximum radial mode number for TM
                for m in range(1, m_max_te + 1):    
                    modes.append(self.TE(n=m, wl=wl))  
                for m in range(1, m_max_tm + 1):
                    modes.append(self.TM(n=m, wl=wl))
            else:   
                m_max = self.m_max(wl=wl, ell=ell) # Maximum radial mode number for hybrid modes
                for m in range(1, m_max + 1):
                    if m % 2 == 1:  # Odd internal m corresponds to HE_{ell,n}, m = 2n - 1
                        n = (m + 1) // 2
                        modes.append(self.HE(ell=ell, n=n, wl=wl))
                    else:           # Even internal m corresponds to EH_{ell,n}, m = 2n
                        n = m // 2
                        modes.append(self.EH(ell=ell, n=n, wl=wl))

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
        return GuidedMode(self, ell=ell, m=m, wl=wl,
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
        return GuidedMode(self, ell=ell, m=m, wl=wl,
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
        return GuidedMode(self, ell=0, m=n, wl=wl,
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
        return GuidedMode(self, ell=0, m=n, wl=wl,
                          mode_type="TM", **kwargs)
    
    def __repr__(self):
        return (f"StepIndexFibre(core_radius={self.core_radius:.3e}, "
                f"eps_core={self.eps_core}, eps_clad={self.eps_clad}, "
                f"mu_core={self.mu_core}, mu_clad={self.mu_clad})")

    # -------------------------- legacy aliases (to be deprecated do not use) --------------------
    wavelength_from_V=wl_of_V
