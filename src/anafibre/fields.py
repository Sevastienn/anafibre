"""
Defines the GuidedMode class, which represents a single eigenmode $(\\ell, m, \\lambda)$ of a step-index fiber.
Now includes a lightweight grid cache to avoid recomputing radial functions and phases when
sampling E/H on the *same* spatial grid repeatedly (e.g., for plotting, power integration,
animations along z, etc.).

Cache design (small & safe):
- Keyed by coordinate system ("xy" or "cyl") and a compact fingerprint of the grid
  (shape, extents, mean step, z if scalar/broadcastable). This avoids storing
  gigantic array hashes and keeps comparisons fast.
- Stored items: flattened rho/phi/z, shape, precomputed radial functions $R_0$, $R_+$, $R_-$,
  and the longitudinal/azimuthal phase $\\exp(i(k_z z + \\ell \\phi))$.
- Invalidation: automatic whenever A/B, $k_z$, $\\ell$, wavelength, fibre, or $V/b$ change,
  i.e., on new GuidedMode instance. You can also call clear_grid_cache().

Notes:
- The cache is *per-mode* (per GuidedMode instance). It never grows beyond a handful of grids
  you touch in a session. Use clear_grid_cache() if you need to free memory.
- If you pass a different grid (different shape or spacing), the cache just recomputes and stores it.

"""

import numpy as np
import warnings
from scipy.special import jv, kv, jvp
from scipy.constants import epsilon_0 as eps0, mu_0 as mu0, c as c0
from .utils import units, _HAS_UNITS, _strip_unit
from .dispersion import b_to_neff, b_to_kz, _wDlnK, Phi_alpha


# ---------------------------- small helpers ---------------------------------

def spin_to_cartesian(F0, Fp, Fm):
    """Convert spin-basis field components to Cartesian components.

    Parameters
    ----------
    F0, Fp, Fm : array-like
        Longitudinal and circular transverse spin components.

    Returns
    -------
    numpy.ndarray
        Array with last axis ``(Fx, Fy, Fz)``.
    """
    Ex = (Fp + Fm) / np.sqrt(2)
    Ey = 1j * (Fp - Fm) / np.sqrt(2)
    Ez = F0
    return np.stack([Ex, Ey, Ez], axis=-1)

def cartesian_to_cylindrical(F_cart, phi):
    """Convert Cartesian vectors to cylindrical components.

    Parameters
    ----------
    F_cart : array-like
        Vector field with last axis ``(Fx, Fy, Fz)``.
    phi : array-like
        Azimuth angle in radians.

    Returns
    -------
    numpy.ndarray
        Vector field with last axis ``(F_rho, F_phi, F_z)``.
    """
    Fx = F_cart[..., 0]
    Fy = F_cart[..., 1]
    Fz = F_cart[..., 2]
    F_rho = Fx * np.cos(phi) + Fy * np.sin(phi)
    F_phi = -Fx * np.sin(phi) + Fy * np.cos(phi)
    return np.stack([F_rho, F_phi, Fz], axis=-1)

def cylindrical_to_cartesian(F_cyl, phi):
    """Convert cylindrical vectors to Cartesian components.

    Parameters
    ----------
    F_cyl : array-like
        Vector field with last axis ``(F_rho, F_phi, F_z)``.
    phi : array-like
        Azimuth angle in radians.

    Returns
    -------
    numpy.ndarray
        Vector field with last axis ``(Fx, Fy, Fz)``.
    """
    F_rho = F_cyl[..., 0]
    F_phi = F_cyl[..., 1]
    Fz = F_cyl[..., 2]
    Fx = F_rho * np.cos(phi) - F_phi * np.sin(phi)
    Fy = F_rho * np.sin(phi) + F_phi * np.cos(phi)
    return np.stack([Fx, Fy, Fz], axis=-1)

def format_complex_auto(z, tol=1e-14):
    """Format a complex scalar (or quantity) with compact zero suppression.

    Parameters
    ----------
    z : complex | astropy.units.Quantity
        Value to format.
    tol : float, default=1e-14
        Threshold below which real/imaginary parts are treated as zero.

    Returns
    -------
    str
        Human-readable complex number string, including unit suffix if present.
    """
    unit_str = ""
    if _HAS_UNITS and hasattr(z, "unit"):
        unit_str = f" {z.unit}"
        z = _strip_unit(z)

    if abs(z.real) < tol:
        return f"{z.imag:.3f} i{unit_str}"
    elif abs(z.imag) < tol:
        return f"{z.real:.3f}{unit_str}"
    else:
        return f"{z.real:.3f} + {z.imag:.3f} i{unit_str}"

class ModeNotFoundError(RuntimeError):
    """Raised when the requested guided mode does not exist."""
    pass

# ---------------------------- core class ------------------------------------

class GuidedMode:
    """Analytical representation of one guided mode at fixed wavelength.

    A ``GuidedMode`` stores the solved modal constants and exposes methods to
    evaluate electric/magnetic phasors and their derivatives on arbitrary grids.
    """
    def __init__(self, fibre, ell, m, wl, mode_type=None, N_b=2000, *, a=1.0, a_plus=None, a_minus=None):
        """
        Parameters
        ----------
        fibre : anafibre.fibre.StepIndexFibre
            Parent fibre object used for material/profile and dispersion queries.
        ell : int
            Azimuthal mode index.
        m : int
            Internal radial mode index (1-based).
        wl : float | Quantity[m]
            Wavelength in meters.
        mode_type : {"HE", "EH", "TE", "TM", None}, optional
            Mode-family selector. For ``ell=0``, use ``"TE"`` or ``"TM"``.
            For ``ell>0``, typically ``"HE"`` or ``"EH"``.
        N_b : int, default=2000
            Number of samples used for root bracketing when solving for ``b``.
        a : complex, default=1.0
            Default superposition coefficient used as ``a_plus`` when
            ``a_plus``/``a_minus`` are not explicitly provided.
        a_plus : complex, optional
            Coefficient of the ``+|ell|`` degenerate partner.
        a_minus : complex, optional
            Coefficient of the ``-|ell|`` degenerate partner. Ignored for
            non-degenerate ``ell=0`` modes.

        Raises
        ------
        anafibre.fields.ModeNotFoundError
            If no guided root exists for the requested ``(ell, m, wl)``.

        Notes
        -----
        If neither ``a_plus`` nor ``a_minus`` is provided, the default is
        ``a_plus=a`` and ``a_minus=0``.
        """
        self.fibre = fibre
        self.ell = ell
        self.m = m
        self.wl = _strip_unit(wl, units.m if _HAS_UNITS else None)
        # self.wavelength = wl
        self.mode_type = mode_type

        # Optional superposition coefficients for ℓ≠0 (degenerate ±|ℓ| pair)
        if (a_plus is None) and (a_minus is None):
            a_plus = a
            a_minus = 0.0
        else:
            # If user is specifying the pair, allow partial specification
            if a_plus is None:
                a_plus = a
            if a_minus is None:
                a_minus = 0.0

        # Handle non-degenerate ℓ = 0 modes (TE/TM etc.)
        if ell == 0 and a_minus != 0:
            warnings.warn(
                "For ℓ=0 modes (non-degenerate), a_minus is ignored.",
                UserWarning,
                stacklevel=2
            )
            a_minus = 0.0

        self.a_plus = complex(a_plus)
        self.a_minus = complex(a_minus)

        self.V = fibre.V(wl)
        self.b = fibre.b(ell, m, wl=self.wl, mode_type=mode_type, N_b=N_b)
        if np.isnan(self.b) or not (0 < self.b < 1):
            raise ModeNotFoundError(
                f"Mode (ℓ={ell}, m={m}) does not exist at λ={wl}."
            )
        self.neff = b_to_neff(fibre, self.b, wl=self.wl)
        self.kz = b_to_kz(fibre, self.b, wl=self.wl)
        self.k0 = 2 * np.pi / _strip_unit(self.wl, units.m if _HAS_UNITS else None)

        self.A, self.B = self._compute_amplitudes()

        # Per-mode grid cache (see module docstring)
        # maps keys -> dict(rho_flat, phi_flat, z_flat, shape, R0, Rp, Rm, phi_phase, s_phase, z_phase)
        self._grid_cache = {}

    # ----------------------- material/geometry wrappers ----------------------
    def eps(self, rho):
        """Evaluate relative permittivity at radial coordinate(s).

        Parameters
        ----------
        rho : float | array-like
            Radial coordinate in meters.

        Returns
        -------
        float | complex | numpy.ndarray
            Relative permittivity.
        """
        return self.fibre.eps(rho, self.wl)

    def mu(self, rho):
        """Evaluate relative permeability at radial coordinate(s).

        Parameters
        ----------
        rho : float | array-like
            Radial coordinate in meters.

        Returns
        -------
        float | complex | numpy.ndarray
            Relative permeability.
        """
        return self.fibre.mu(rho, self.wl)

    def n(self, rho):
        """Evaluate refractive index at radial coordinate(s).

        Parameters
        ----------
        rho : float | array-like
            Radial coordinate in meters.

        Returns
        -------
        float | complex | numpy.ndarray
            Refractive index.
        """
        return self.fibre.n(rho, self.wl)

    def k(self, rho):
        """Evaluate local wave number ``k = k0 * n(rho)``.

        Parameters
        ----------
        rho : float | array-like
            Radial coordinate in meters.

        Returns
        -------
        float | complex | numpy.ndarray
            Local wave number in rad/m.
        """
        return self.k0 * self.n(rho)

    def kap(self, rho):
        """Evaluate local transverse wave number.

        Parameters
        ----------
        rho : float | array-like
            Radial coordinate in meters.

        Returns
        -------
        complex | numpy.ndarray
            ``sqrt(k(rho)^2 - kz^2)`` with complex branch support.
        """
        k = self.k(rho)
        return np.sqrt(k**2 - self.kz**2 + 0j)

    # -------------------------- cache management -----------------------------
    def clear_grid_cache(self):
        """Clear cached grid-dependent intermediate arrays.

        Returns
        -------
        None
            Empties the per-instance grid cache.
        """
        self._grid_cache.clear()

    @staticmethod
    def _fingerprint_axis(arr):
        """Return a small numeric fingerprint for an axis array (1D or from a 2D mesh).
        Uses shape, min/max, and mean step (to within a tolerance) to characterise a grid.
        """
        a = np.asarray(arr)
        if a.ndim == 2:
            # If it's from meshgrid, prefer the varying direction for step estimation
            if a.shape[0] > 1:
                steps = np.diff(a[:, 0])
            else:
                steps = np.diff(a[0, :])
        elif a.ndim == 1:
            steps = np.diff(a)
        else:
            steps = np.array([0.0])

        step = float(np.nanmean(steps)) if steps.size else 0.0
        return (a.shape, float(np.nanmin(a)), float(np.nanmax(a)), step)

    def _grid_key(self, *, coord, rho=None, phi=None, z=0, x=None, y=None):
        """Compose a compact, hashable key describing the provided grid."""
        if coord == "xy":
            fx = self._fingerprint_axis(x)
            fy = self._fingerprint_axis(y)
            # z can be array-like; use its broadcasted scalar if possible
            z_val = np.asarray(z)
            z_fp = float(np.nanmean(z_val)) if z_val.size else float(z_val)
            return ("xy", fx, fy, round(z_fp, 12))
        else:  # cylindrical
            fr = self._fingerprint_axis(rho)
            fphi = self._fingerprint_axis(phi)
            z_val = np.asarray(z)
            z_fp = float(np.nanmean(z_val)) if z_val.size else float(z_val)
            return ("cyl", fr, fphi, round(z_fp, 12))

    # -------------------------- amplitudes/normalisation ---------------------
    def sigma(self, A=None, B=None):
        """Compute power-normalization coefficient for modal amplitudes.

        Parameters
        ----------
        A, B : complex, optional
            Longitudinal-field coefficients. Defaults to the instance values.

        Returns
        -------
        complex | float
            Modal normalization coefficient used in amplitude normalization.
        """
        fibre = self.fibre
        ell = int(abs(self.ell))
        V = self.V
        wl = self.wl
        if A is None:
            A = self.A
        if B is None:
            B = self.B
        k0 = self.k0
        kz = self.kz
        a = fibre.core_radius

        eps1 = fibre._eval(fibre.eps_core, wl)
        eps2 = fibre._eval(fibre.eps_clad, wl)
        mu1 = fibre._eval(fibre.mu_core, wl)
        mu2 = fibre._eval(fibre.mu_clad, wl)

        n1 = fibre.n_core(wl)
        n2 = fibre.n_clad(wl)
        k1 = k0 * n1
        k2 = k0 * n2

        u = V * np.sqrt(1 - self.b)
        w = V * np.sqrt(self.b)

        if A == 0 and B == 0:
            return 1.0

        DlnJ = jvp(ell, u)/jv(ell, u)
        I1_plus = 1/2 - ell**2 / (2 * u**2) + DlnJ / u + DlnJ**2 / 2
        I1_minus = ell / (u**2)

        DlnK = _wDlnK(ell, w)/w
        I2_plus = 1/2 + ell**2 / (2 * w**2) - DlnK / w - DlnK**2 / 2
        I2_minus = -ell / (w**2)

        alpha1_plus = (eps1 * np.abs(A) ** 2 + mu1 * np.abs(B) ** 2) / (np.abs(A) ** 2 + np.abs(B) ** 2)
        alpha2_plus = (eps2 * np.abs(A) ** 2 + mu2 * np.abs(B) ** 2) / (np.abs(A) ** 2 + np.abs(B) ** 2)
        alpha_minus = (np.imag(A * np.conj(B))) / (np.abs(A)**2 + np.abs(B)**2)

        term1 = (kz * k0) / u**2 * alpha1_plus * I1_plus
        term2 = (kz**2 + k1**2) / u**2 * alpha_minus * I1_minus
        term3 = (kz * k0) / w**2 * alpha2_plus * I2_plus
        term4 = (kz**2 + k2**2) / w**2 * alpha_minus * I2_minus

        sigma = np.pi * a**4 * (term1 + term2 + term3 + term4)

        return sigma

    def _compute_amplitudes(self, Normalise=True):
        fibre = self.fibre
        ell = self.ell
        b = self.b
        V = self.V
        wl = self.wl
        ne = self.neff

        eps1 = fibre._eval(fibre.eps_core, wl)
        eps2 = fibre._eval(fibre.eps_clad, wl)
        mu1 = fibre._eval(fibre.mu_core, wl)
        mu2 = fibre._eval(fibre.mu_clad, wl)
        phi_eps = Phi_alpha(ell=ell, b=b, V=V, alpha=[eps1, eps2])
        phi_mu = Phi_alpha(ell=ell, b=b, V=V, alpha=[mu1, mu2])

        if ell == 0:
            mt = self.mode_type.lower() if self.mode_type is not None else None
            if mt is None:
                is_te = np.isclose(phi_mu, 0, atol=1e-9)
                is_tm = np.isclose(phi_eps, 0, atol=1e-9)
                if is_te and not is_tm:
                    mt = "te"; phi_mu = 0*phi_eps
                elif is_tm and not is_te:
                    mt = "tm"; phi_eps = 0*phi_mu
            if mt == "te":
                A, B = 0.0, 1.0j
            elif mt == "tm":
                A, B = 1.0, 0.0
            else:
                raise ValueError("mode_type must be 'TE', 'TM', or None for ℓ = 0.")
        else:  # ℓ ≠ 0 hybrid
            nu_eps = 1j * phi_eps / (ell * ne)
            nu = np.sqrt(-phi_eps / phi_mu + 0j)
            if np.real(nu * np.conj(nu_eps)) < 0:
                nu = -nu
            A = 1 / np.sqrt(1 + np.abs(nu)**2)
            B = nu * A

        N = 1/np.sqrt(c0 * self.sigma(A, B)) if Normalise else 1.0
        return N*A, N*B

    # --------------------------- radial functions ----------------------------
    def _radial_function(self, s, rho):
        rho0 = self.fibre.core_radius

        k1 = self.k0 * self.fibre.n_core(self.wl)
        k2 = self.k0 * self.fibre.n_clad(self.wl)
        kz = self.kz

        if np.iscomplex(kz):
            kap1 = np.sqrt(k1**2 - kz**2 + 0j)
            gam2 = np.sqrt(kz**2 - k2**2 + 0j)
        else:
            kap1 = np.sqrt(k1**2 - kz**2)
            gam2 = np.sqrt(kz**2 - k2**2)

        inside = rho < rho0
        R = np.empty_like(rho, dtype=complex)
        R[inside] = jv(self.ell - s, kap1 * rho[inside]) / jv(self.ell, kap1 * rho0)
        R[~inside] = (1j)**(s) * kv(self.ell - s, gam2 * rho[~inside]) / kv(self.ell, gam2 * rho0)
        return R

    # --------------------------- cache-aware backend -------------------------
    def _prepare_grid(self, *, rho=None, phi=None, z=0, x=None, y=None):
        """Return (rho_flat, phi_flat, z_flat, shape, phi_phase, s_phase, z_phase, R0, Rp, Rm) using cache if possible."""
        if (x is not None) and (y is not None):
            coord = "xy"
            x = np.asarray(x); y = np.asarray(y)
            rho_arr = np.sqrt(x**2 + y**2)
            phi_arr = np.arctan2(y, x)
            shape = np.broadcast_shapes(x.shape, y.shape)
        else:
            coord = "cyl"
            if rho is None or phi is None:
                raise ValueError("Provide either (rho, phi) or (x, y) as input.")
            rho_arr = np.asarray(rho)
            phi_arr = np.asarray(phi)
            shape = np.broadcast_shapes(rho_arr.shape, phi_arr.shape)

        z_arr = np.asarray(z)
        rho_b, phi_b, z_b = np.broadcast_arrays(rho_arr, phi_arr, z_arr)
        key = self._grid_key(coord=coord, rho=rho_b, phi=phi_b, z=z_b, x=x, y=y)

        cache = self._grid_cache.get(key)
        if cache is None:
            # compute and store
            rho_f = rho_b.ravel()
            phi_f = phi_b.ravel()
            z_f = z_b.ravel()

            R0 = self._radial_function(0, rho_f)
            Rp = self._radial_function(+1, rho_f)
            Rm = self._radial_function(-1, rho_f)
            phi_phase = np.exp(1j * self.ell * phi_f)
            s_phase = np.exp(1j * phi_f)
            z_phase = np.exp(1j * self.kz * z_f)

            cache = {
                "rho_flat": R0*0 + rho_f,  # ensure same shape/dtype
                "phi_flat": R0*0 + phi_f,
                "z_flat": R0*0 + z_f,
                "shape": shape,
                "R0": R0,
                "Rp": Rp,
                "Rm": Rm,
                "phi_phase": phi_phase,
                "s_phase": s_phase,
                "z_phase": z_phase,
            }
            self._grid_cache[key] = cache

        return (
            cache["rho_flat"], cache["phi_flat"], cache["z_flat"], cache["shape"],
            cache["phi_phase"], cache["s_phase"], cache["z_phase"], cache["R0"], cache["Rp"], cache["Rm"],
        )

    def _spin_components(self, field, rho_flat, phi_phase, s_phase, z_phase):
        A, B = self.A, self.B
        kz = self.kz
        k0 = self.k0

        if field.lower() == 'e':
            A1, A2 = A, B
            alpha = self.mu(rho_flat)
            beta0 = eps0
        elif field.lower() == 'h':
            A1, A2 = B, -A
            alpha = self.eps(rho_flat)
            beta0 = mu0
        else:
            raise ValueError("field must be 'E' or 'H'")

        kap = self.kap(rho_flat)
        # Use precomputed radial functions from caller when possible
        # (They are passed in by E/H via _prepare_grid.)
        # This helper expects the caller to multiply with Rp/Rm/R0 externally
        F0 = (A1/np.sqrt(beta0))
        c_plus = ((+1j * kz * A1 - k0 * alpha * A2) / (kap * np.sqrt(2*beta0)))
        c_minus = ((-1j * kz * A1 - k0 * alpha * A2) / (kap * np.sqrt(2*beta0)))

        return F0, c_plus, c_minus, phi_phase, s_phase, z_phase
    
    def _radial_log_derivative(self, s, ell, rho):
        """
        d/dρ ln R_s(ρ) consistent with _radial_function(s, ρ).

        IMPORTANT: rho may be complex dtype due to caching trick (R0*0 + rho).
        We use rho_r = real(rho) for masks and geometry.
        """
        rho_r = np.asarray(np.real(rho), dtype=float)

        rho0 = self.fibre.core_radius
        k1 = self.k0 * self.fibre.n_core(self.wl)
        k2 = self.k0 * self.fibre.n_clad(self.wl)
        kz = self.kz

        if np.iscomplex(kz):
            kap1 = np.sqrt(k1**2 - kz**2 + 0j)
            gam2 = np.sqrt(kz**2 - k2**2 + 0j)
        else:
            kap1 = np.sqrt(k1**2 - kz**2)
            gam2 = np.sqrt(kz**2 - k2**2)

        nu = ell - s
        inside = rho_r < rho0

        out = np.empty_like(rho_r, dtype=complex)

        # Core: R = J_{nu}(kap1 ρ) / J_{ell}(kap1 ρ0) -> d ln R / dρ = kap1 * J' / J
        if np.any(inside):
            x = kap1 * rho_r[inside]
            with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
                out[inside] = kap1 * (jvp(nu, x) / jv(nu, x))

        # Cladding: R = (1j)^s K_{nu}(gam2 ρ) / K_{ell}(gam2 ρ0) -> d ln R / dρ = gam2 * K' / K
        # Use stable weighted derivative: _wDlnK(nu, x) = x K'/K -> K'/K = _wDlnK/x
        if np.any(~inside):
            x = gam2 * rho_r[~inside]
            with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
                out[~inside] = gam2 * (_wDlnK(nu, x) / x)

        return out

    def _field(self, which, rho=None, phi=None, z=0, *, x=None, y=None, coord="cartesian"):
        """Evaluate E/H phasor with optional Cartesian or cylindrical output."""
        which = str(which).upper()
        if which not in ("E", "H"):
            raise ValueError("which must be 'E' or 'H'.")

        (rho_f, phi_f, _z_f, shape, phi_phase, s_phase, z_phase, R0, Rp, Rm) = self._prepare_grid(
            rho=rho, phi=phi, z=z, x=x, y=y
        )
        F0, c_plus, c_minus, phi_phase, s_phase, z_phase = self._spin_components(
            which, rho_f, phi_phase, s_phase, z_phase
        )

        F0_plus = F0 * R0 * phi_phase * z_phase
        Fp_plus = c_plus * Rp * np.conj(s_phase) * phi_phase * z_phase
        Fm_plus = c_minus * Rm * s_phase * phi_phase * z_phase

        minus_sign = -1 if which == "H" else 1
        F0_minus = minus_sign * F0 * R0 * np.conj(phi_phase) * z_phase
        Fp_minus = minus_sign * c_minus * Rm * np.conj(s_phase) * np.conj(phi_phase) * z_phase
        Fm_minus = minus_sign * c_plus * Rp * s_phase * np.conj(phi_phase) * z_phase

        F0 = self.a_plus * F0_plus + self.a_minus * F0_minus
        Fp = self.a_plus * Fp_plus + self.a_minus * Fp_minus
        Fm = self.a_plus * Fm_plus + self.a_minus * Fm_minus

        F_cart = spin_to_cartesian(F0, Fp, Fm).reshape(*shape, 3)

        coord = str(coord).lower()
        if coord.startswith("cart") or coord.startswith("car"):
            return F_cart
        if coord.startswith("cyl"):
            phi_r = np.asarray(np.real(phi_f), dtype=float).reshape(shape)
            return cartesian_to_cylindrical(F_cart, phi_r)
        raise ValueError("coord must be 'cartesian' or 'cylindrical'.")

    def _grad_field(self, which, rho=None, phi=None, z=0, *, x=None, y=None, coord="cartesian"):
        """
        Analytical Jacobian for E or H.

        Returns J[..., i, j] = ∂_j F_i in Cartesian components i=(x,y,z).
        If coord='cylindrical', columns are (∂/∂ρ, (1/ρ)∂/∂φ, ∂/∂z).
        """
        which = str(which).upper()
        if which not in ("E", "H"):
            raise ValueError("which must be 'E' or 'H'.")

        (rho_f, phi_f, z_f, shape, phi_phase, s_phase, z_phase, R0, Rp, Rm) = self._prepare_grid(
            rho=rho, phi=phi, z=z, x=x, y=y
        )

        # rho_f is complex dtype (cache trick). Use real for geometry.
        rho_r = np.asarray(np.real(rho_f), dtype=float)
        phi_r = np.asarray(np.real(phi_f), dtype=float)

        # Spin coefficients (exactly as your E/H do)
        F0c, c_plus, c_minus, phi_phase, s_phase, z_phase = self._spin_components(
            which, rho_f, phi_phase, s_phase, z_phase
        )

        # Build branch-resolved spin fields exactly like E()/H()
        # "+" branch
        F0_plus = F0c * R0 * phi_phase * z_phase
        Fp_plus = c_plus  * Rp * np.conj(s_phase) * phi_phase * z_phase     # exp(i(ℓ-1)φ)
        Fm_plus = c_minus * Rm * s_phase          * phi_phase * z_phase     # exp(i(ℓ+1)φ)

        # "-" branch (differs by sign between E and H exactly as in your code)
        if which == "E":
            F0_minus =  F0c    * R0 * np.conj(phi_phase) * z_phase
            Fp_minus =  c_minus * Rm * np.conj(s_phase) * np.conj(phi_phase) * z_phase  # exp(-i(ℓ+1)φ)
            Fm_minus =  c_plus  * Rp * s_phase          * np.conj(phi_phase) * z_phase  # exp(-i(ℓ-1)φ)
        else:
            F0_minus = -F0c    * R0 * np.conj(phi_phase) * z_phase
            Fp_minus = -c_minus * Rm * np.conj(s_phase) * np.conj(phi_phase) * z_phase
            Fm_minus = -c_plus  * Rp * s_phase          * np.conj(phi_phase) * z_phase

        # Superpose polarisation coefficients
        F0 = self.a_plus * F0_plus + self.a_minus * F0_minus
        Fp = self.a_plus * Fp_plus + self.a_minus * Fp_minus
        Fm = self.a_plus * Fm_plus + self.a_minus * Fm_minus

        # --- ρ-derivatives: only radial functions depend on ρ (within each region) ---
        dlnR0p = self._radial_log_derivative(0, self.ell, rho_f)
        dlnRpp = self._radial_log_derivative(+1, self.ell, rho_f)
        dlnRmp = self._radial_log_derivative(-1, self.ell, rho_f)
        dlnR0l = self._radial_log_derivative(0, -self.ell, rho_f)
        dlnRpl = self._radial_log_derivative(+1, -self.ell, rho_f)
        dlnRml = self._radial_log_derivative(-1, -self.ell, rho_f)

        dρF0 = self.a_plus * dlnR0p * F0_plus + self.a_minus * dlnR0l * F0_minus
        dρFp = self.a_plus * dlnRpp * Fp_plus + self.a_minus * dlnRpl * Fp_minus
        dρFm = self.a_plus * dlnRmp * Fm_plus + self.a_minus * dlnRml * Fm_minus

        # --- φ-derivatives: MUST be branch-wise with correct exponents ---
        ell = self.ell

        # plus branch exponents:
        # F0_plus ~ exp(+i ell φ)
        # Fp_plus ~ exp(+i(ell-1)φ)
        # Fm_plus ~ exp(+i(ell+1)φ)
        # minus branch exponents:
        # F0_minus ~ exp(-i ell φ)
        # Fp_minus ~ exp(-i(ell+1)φ)
        # Fm_minus ~ exp(-i(ell-1)φ)
        dφF0 = 1j * ( self.a_plus * (+ell)       * F0_plus + self.a_minus * (-ell)        * F0_minus )
        dφFp = 1j * ( self.a_plus * (ell - 1)    * Fp_plus + self.a_minus * (-(ell + 1))  * Fp_minus )
        dφFm = 1j * ( self.a_plus * (ell + 1)    * Fm_plus + self.a_minus * (-(ell - 1))  * Fm_minus )

        # --- z-derivatives ---
        dz_fac = 1j * self.kz
        dzF0 = dz_fac * F0
        dzFp = dz_fac * Fp
        dzFm = dz_fac * Fm

        # Convert spin partials -> Cartesian partial vectors
        dρ_cart = spin_to_cartesian(dρF0, dρFp, dρFm)   # (N,3)
        dφ_cart = spin_to_cartesian(dφF0, dφFp, dφFm)
        dz_cart = spin_to_cartesian(dzF0, dzFp, dzFm)

        coord = str(coord).lower()
        if coord.startswith("cyl"):
            invρ = np.zeros_like(rho_r, dtype=float)
            m = rho_r > 0
            invρ[m] = 1.0 / rho_r[m]
            # columns = (∂/∂ρ, (1/ρ)∂/∂φ, ∂/∂z)
            J = np.stack([dρ_cart, invρ[:, None] * dφ_cart, dz_cart], axis=-1)
            return J.reshape(*shape, 3, 3)

        # Cartesian chain rule:
        # ∂x = cosφ ∂ρ - (sinφ/ρ) ∂φ
        # ∂y = sinφ ∂ρ + (cosφ/ρ) ∂φ
        c = np.cos(phi_r)
        s = np.sin(phi_r)

        invρ = np.zeros_like(rho_r, dtype=float)
        m = rho_r > 0
        invρ[m] = 1.0 / rho_r[m]

        dX = c[:, None] * dρ_cart - (s * invρ)[:, None] * dφ_cart
        dY = s[:, None] * dρ_cart + (c * invρ)[:, None] * dφ_cart
        dZ = dz_cart

        J = np.stack([dX, dY, dZ], axis=-1)
        return J.reshape(*shape, 3, 3)

    # ------------------------------- public API ------------------------------
    def E(self, rho=None, phi=None, z=0, *, x=None, y=None, coord="cartesian"):
        """Evaluate electric field phasor on a transverse grid.

        Parameters
        ----------
        rho, phi : array-like, optional
            Cylindrical coordinates in meters/radians.
        z : float | array-like, default=0
            Longitudinal position in meters.
        x, y : array-like, optional
            Cartesian coordinates in meters.
        coord : {"cartesian", "cylindrical"}, default="cartesian"
            Output vector basis. Also accepts ``"cart"``/``"cyl"`` prefixes.

        Returns
        -------
        numpy.ndarray
            Complex electric field with last axis ``(Ex, Ey, Ez)`` for Cartesian,
            or ``(E_rho, E_phi, E_z)`` for cylindrical output.

        Notes
        -----
        The returned phasor is built from the cylindrical-mode expansion
        $\\sum_s E^{(s)}(\\rho)\\,\\mathbf{e}_s\\,e^{i(\\ell-s)\\phi}e^{ik_z z}$, with optional
        superposition of degenerate ``+/-ell`` partners via ``a_plus`` and
        ``a_minus``.
        """
        return self._field("E", rho=rho, phi=phi, z=z, x=x, y=y, coord=coord)
    
    def H(self, rho=None, phi=None, z=0, *, x=None, y=None, coord="cartesian"):
        """Evaluate magnetic field phasor on a transverse grid.

        Parameters
        ----------
        rho, phi : array-like, optional
            Cylindrical coordinates in meters/radians.
        z : float | array-like, default=0
            Longitudinal position in meters.
        x, y : array-like, optional
            Cartesian coordinates in meters.
        coord : {"cartesian", "cylindrical"}, default="cartesian"
            Output vector basis. Also accepts ``"cart"``/``"cyl"`` prefixes.

        Returns
        -------
        numpy.ndarray
            Complex magnetic field with last axis ``(Hx, Hy, Hz)`` for Cartesian,
            or ``(H_rho, H_phi, H_z)`` for cylindrical output.

        Notes
        -----
        Uses the same modal ansatz and phase convention as :meth:`E`, with magnetic
        spin components linked to electric ones through Maxwell-coupled amplitudes.
        """
        return self._field("H", rho=rho, phi=phi, z=z, x=x, y=y, coord=coord)
    
    def gradE(self, rho=None, phi=None, z=0, *, x=None, y=None, coord="cartesian"):
        """Evaluate spatial Jacobian of the electric field.

        Parameters
        ----------
        rho, phi : array-like, optional
            Cylindrical coordinates in meters/radians.
        z : float | array-like, default=0
            Longitudinal position in meters.
        x, y : array-like, optional
            Cartesian coordinates in meters.
        coord : {"cartesian", "cylindrical"}, default="cartesian"
            Coordinate basis for derivative columns.

        Returns
        -------
        numpy.ndarray
            Array ``J[..., i, j] = dE_i / dx_j``.

        Notes
        -----
        Implements the analytical Jacobian derived from the mode ansatz. For
        ``coord='cylindrical'``, derivative columns are ordered as
        $(\\partial/\\partial\\rho,\\,(1/\\rho)\\partial/\\partial\\phi,\\,\\partial/\\partial z)$.
        """
        return self._grad_field("E", rho=rho, phi=phi, z=z, x=x, y=y, coord=coord)

    def gradH(self, rho=None, phi=None, z=0, *, x=None, y=None, coord="cartesian"):
        """Evaluate spatial Jacobian of the magnetic field.

        Parameters
        ----------
        rho, phi : array-like, optional
            Cylindrical coordinates in meters/radians.
        z : float | array-like, default=0
            Longitudinal position in meters.
        x, y : array-like, optional
            Cartesian coordinates in meters.
        coord : {"cartesian", "cylindrical"}, default="cartesian"
            Coordinate basis for derivative columns.

        Returns
        -------
        numpy.ndarray
            Array ``J[..., i, j] = dH_i / dx_j``.

        Notes
        -----
        Magnetic-field Jacobian counterpart of :meth:`gradE`, using the same
        analytical derivative pathway (no finite differences).
        """
        return self._grad_field("H", rho=rho, phi=phi, z=z, x=x, y=y, coord=coord)
    
    def Power(self, rho=None, phi=None, z=0, *, x=None, y=None,
            auto=None, extent_factor=8.0, N=1000, tol=1e-3, max_iter=8,
            expand_factor=1.5, cache=True, geometry="square"):
        """
        Compute guided power by integrating the time-averaged Poynting flux.

        Parameters
        ----------
        rho, phi : array-like, optional
            Polar integration grid in meters/radians.
        z : float, default=0
            Longitudinal position in meters.
        x, y : array-like, optional
            Cartesian integration grid in meters.
        auto : bool | None, default=None
            If ``True`` (or ``None`` with no grid given), run adaptive square-grid
            integration. If ``False``, explicit grid input is required.
        extent_factor : float, default=8.0
            Initial half-width in units of core radius for adaptive integration.
        N : int, default=1000
            Grid resolution per axis for adaptive integration.
        tol : float, default=1e-3
            Relative convergence threshold for adaptive integration.
        max_iter : int, default=8
            Maximum number of domain expansions in adaptive mode.
        expand_factor : float, default=1.5
            Domain growth multiplier between adaptive iterations.
        cache : bool, default=True
            Cache the latest adaptive-integration result.
        geometry : {"square"}, default="square"
            Adaptive integration geometry.

        Returns
        -------
        float
            Estimated guided power crossing the transverse plane.

        Notes
        -----
        Computes the time-averaged power
        $P=\\frac{1}{2}\\int \\mathrm{Re}(\\mathbf{E}\\times\\mathbf{H}^*)\\cdot\\hat{\\mathbf{z}}\\,dA$
        over the transverse plane. This is the numerical counterpart of the
        analytical normalization relation used for $\\sigma$ in the paper.
        """
        # --- 1) Manual Cartesian grid (existing path) ---
        if x is not None or y is not None:
            if x is None or y is None:
                raise ValueError("Provide both x and y.")
            x = np.asarray(x); y = np.asarray(y)

            E = self.E(x=x, y=y, z=z)
            H = self.H(x=x, y=y, z=z)
            S = 0.5 * np.real(np.cross(E, np.conj(H)))  # (.., 3)

            dx = np.mean(np.diff(x[0, :])) if x.ndim == 2 else np.mean(np.diff(x))
            dy = np.mean(np.diff(y[:, 0])) if y.ndim == 2 else np.mean(np.diff(y))
            dA = dx * dy

            P = np.sum(S[..., 2]) * dA
            return P

        # --- 2) Manual polar grid (existing path) ---
        if rho is not None or phi is not None:
            if rho is None or phi is None:
                raise ValueError("Provide both rho and phi.")
            rho = np.asarray(rho); phi = np.asarray(phi)

            E = self.E(rho=rho, phi=phi, z=z)
            H = self.H(rho=rho, phi=phi, z=z)
            S = 0.5 * np.real(np.cross(E, np.conj(H)))

            # crude average area weight in polar sampling (assumes near-uniform grids)
            # Note: this assumes rho and phi are 2D meshgrids with rho varying along axis=1 and phi along axis=0
            drho = rho[0, 1] - rho[0, 0]
            dphi = phi[1, 0] - phi[0, 0]
            dA = (rho * drho * dphi)

            P = np.sum(S[..., 2] * dA)
            return P

        # --- 3) No transverse grid provided: auto mode ---
        if auto is False:
            raise ValueError("No (x,y) or (rho,phi) provided and auto=False.")
        if auto is None:
            auto = True

        if not auto:
            raise ValueError("Provide either (x, y) or (rho, phi) as input.")

        if geometry != "square":
            raise ValueError(f"Auto geometry '{geometry}' not supported (only 'square').")

        # --- 4) Caching (recommended) ---
        if cache:
            key = (float(z), float(extent_factor), int(N), float(tol), int(max_iter), float(expand_factor), str(geometry))
            cache_dict = getattr(self, "_power_cache", None)
            if isinstance(cache_dict, dict) and cache_dict.get("key") == key:
                return cache_dict["P"]

        # --- 5) Determine core radius scale ---
        # Try common attribute names; adjust if your class uses something else.
        a = getattr(getattr(self, "fibre", None), "core_radius", None)
        if a is None:
            a = getattr(self, "core_radius", None)
        if a is None:
            raise AttributeError("Cannot auto-select grid: core radius not found (expected self.fibre.core_radius).")

        # Handle astropy Quantity or plain float
        try:
            a_m = a.to_value("m")
        except Exception:
            a_m = float(a)

        L = extent_factor * a_m

        # --- 6) Expanding-grid integration until convergence ---
        P_prev = None
        last_rel = None

        for it in range(max_iter):
            xs = np.linspace(-L, L, N)
            ys = np.linspace(-L, L, N)
            X, Y = np.meshgrid(xs, ys, indexing="xy")

            # Call the same function on explicit grid to avoid duplicating logic:
            P = self.Power(x=X, y=Y, z=z, auto=False, cache=False)

            if P_prev is not None:
                last_rel = abs(P - P_prev) / (abs(P) + 1e-300)
                if last_rel < tol:
                    break

            P_prev = P
            L *= expand_factor

        else:
            # loop exhausted without break
            warnings.warn(
                f"Power auto-integration did not converge within max_iter={max_iter}. "
                f"Last relative change was {last_rel}. Consider increasing N, extent_factor, or max_iter."
            )

        if cache:
            self._power_cache = {"key": key, "P": P, "L": L, "N": N, "tol": tol, "max_iter": max_iter}

        return P

    # ------------------------------ labelling --------------------------------
    @property
    def mode_label(self):
        """Return HTML-formatted mode family label.

        Returns
        -------
        str
            Label such as ``HE<sub>11</sub>`` or ``TE<sub>01</sub>``.
        """
        if self.ell == 0:
            mt = (self.mode_type or "").upper()
            if mt not in {"TE", "TM"}:
                mt = "TE" if abs(self.A) < 1e-15 else ("TM" if abs(self.B) < 1e-15 else "TE")
                n = self._radial_n()
            n = self.m
            return f"{mt}<sub>0{n}</sub>"
        else:
            mt = (self.mode_type or "").upper()
            if mt not in {"HE", "EH"}:
                mt = self._mode_kind_hybrid()
                n = self._radial_n()
            n = self._radial_n()
            return f"{mt}<sub>{self.ell}{n}</sub>"
        
    def _mode_kind_hybrid(self):
        A, B = self.A, self.B
        s = np.sign(self.ell) * np.imag(A * np.conj(B))
        if s > 0:
            return "HE"
        elif s < 0:
            return "EH"
        else:
            return "HE" if (self.m % 2 == 1) else "EH"

    def _radial_n(self):
        if self.ell == 0:
            mt = (self.mode_type or "").upper()
            if mt not in {"TE", "TM"}:
                return (self.m + 1) // 2
            else:
                return self.m
        return (self.m + 1) // 2

    @property
    def S0(self):
        """Unnormalised Stokes-like intensity parameter."""
        return np.abs(self.a_plus) ** 2 + np.abs(self.a_minus) ** 2

    def _stokes_raw_from_coeffs(self):
        S0 = np.abs(self.a_plus) ** 2 + np.abs(self.a_minus) ** 2
        S1 = 2 * np.real(self.a_plus * np.conj(self.a_minus))
        S2 = 2 * np.imag(self.a_plus * np.conj(self.a_minus))
        S3 = np.abs(self.a_plus) ** 2 - np.abs(self.a_minus) ** 2
        return S0, S1, S2, S3

    def _stokes_normalised_components(self):
        S0, S1_raw, S2_raw, S3_raw = self._stokes_raw_from_coeffs()
        if self.ell == 0:
            mt = (self.mode_type or "").upper()
            if mt == "TE":
                return -1.0, 0.0, 0.0
            if mt == "TM":
                return 1.0, 0.0, 0.0
        if np.isclose(S0, 0.0):
            return np.nan, np.nan, np.nan
        return S1_raw / S0, S2_raw / S0, S3_raw / S0

    @property
    def S1(self):
        """Unnormalised Stokes-like parameter ``S0 * s1``."""
        s1, _, _ = self._stokes_normalised_components()
        return self.S0 * s1

    @property
    def S2(self):
        """Unnormalised Stokes-like parameter ``S0 * s2``."""
        _, s2, _ = self._stokes_normalised_components()
        return self.S0 * s2

    @property
    def S3(self):
        """Unnormalised Stokes-like parameter ``S0 * s3``."""
        _, _, s3 = self._stokes_normalised_components()
        return self.S0 * s3

    def stokes(self, normalised=False):
        """Return Stokes-like parameters derived from ``a_plus/a_minus``."""
        S0 = self.S0
        s1, s2, s3 = self._stokes_normalised_components()
        if not normalised:
            return S0, S0 * s1, S0 * s2, S0 * s3
        return S0, s1, s2, s3

    def __repr__(self):
        S0, S1, S2, S3 = self.stokes(normalised=True)
        return (
            f"<GuidedMode {self.mode_type} (ℓ={self.ell}, n={self.m}), "
            f"λ={self.wl:.2e}, V={self.V:.2f}, neff={self.neff:.4f}, (S₁,S₂,S₃) = {S0:.2f} · ({S1:.2f}, {S2:.2f}, {S3:.2f})>"
            # f"A={self.A:.3f}, B={self.B:.3f}>"
        )
    # ------------------------------ rich display -----------------------------
    def _repr_html_(self):
        from .utils import _HAS_UNITS, wavelength_to_rgb, wavelength_band_label_nm
        try:
            wl_nm = self.wl.to_value('nm') if _HAS_UNITS else float(self.wl) * 1e9
        except Exception:
            wl_nm = float(self.wl) * 1e9

        def _swatch(bg_css, label=None):
            text = (label or "")
            title = f' title="{label}"' if label else ""
            return (
                f'<span{title} style="display:inline-flex;align-items:center;justify-content:center;'
                'width:1.5em;height:1.5em;margin-right:.4em;margin-bottom:0.3em;'
                'border:1px solid rgba(0,0,0,.25);border-radius:2px;'
                f'background:{bg_css};color:white;'
                'font-size:clamp(0.45em, 0.55em, 0.65em);'
                'font-weight:700;letter-spacing:.02em;vertical-align:middle;'
                'overflow:hidden;white-space:nowrap;text-overflow:ellipsis;">'
                f'{text}'
                '</span>'
            )

        label, gray = wavelength_band_label_nm(wl_nm) if wl_nm is not None else (None, None)
        if wl_nm is not None and 380 <= wl_nm <= 780:
            R, G, B = wavelength_to_rgb(wl_nm)
            swatch_html = _swatch(f"rgb({int(R*255)}, {int(G*255)}, {int(B*255)})")
        elif label is not None:
            swatch_html = _swatch(gray, label)
        else:
            swatch_html = ""

        S0, S1, S2, S3 = self.stokes(normalised=True)
        def fmt_signed(x):
            return f"{x:+.2f}".replace("+", "\u2008")

        html = f"""
        <div class="anafibre-guidedmode" aria-label="Guided mode summary"
            style="
                color-scheme: light dark;
                --bg: Canvas;               /* theme background */
                --fg: CanvasText;           /* theme text */
                --border: color-mix(in srgb, var(--fg) 18%, var(--bg));
                --header: color-mix(in srgb, var(--fg) 8%,  var(--bg));
                --row: color-mix(in srgb, var(--fg) 4%,  var(--bg));
                font-variant-numeric: tabular-nums;
            ">
        <style>
            .anafibre-guidedmode table {{
            width: auto;
            background: var(--bg);
            color: var(--fg);
            border: 1px solid var(--border);
            border-radius: 10px;
            border-collapse: separate;
            border-spacing: 0;
            overflow: hidden;
            box-shadow: 0 2px 10px color-mix(in srgb, var(--fg) 10%, transparent);
            }}
            .anafibre-guidedmode th, .anafibre-guidedmode td {{
            padding: .55rem .5rem;
            vertical-align: middle;
            }}
            .anafibre-guidedmode thead th {{
            background: var(--header);
            font-weight: 600;
            }}
            .anafibre-guidedmode td.num, .anafibre-guidedmode th.num {{ text-align: center; }}
            .anafibre-guidedmode tbody tr:nth-child(odd) td {{ background: var(--row); }}
        </style>

        <table>
            <thead>
            <tr>
                <th scope="col" style="text-align:center;">Mode</th>
                <th scope="col" class="num"><i>λ</i> [nm]</th>
                <th scope="col" class="num"><i>V</i></th>
                <th scope="col" class="num"><i>n</i><sub>eff</sub></th>
                <th scope="col" class="num"><i>S</i><sub>0</sub></th>
                <th scope="col" class="num">(<i>S</i><sub>1</sub>, <i>S</i><sub>2</sub>, <i>S</i><sub>3</sub>) / <i>S</i><sub>0</sub></th>
            </tr>
            </thead>
            <tbody>
            <tr>
                <td style="text-align:center;">{self.mode_label}</td>
                <!-- <td style="text-align:center;">{self.ell}</td>
                <td style="text-align:center;">{self.m}</td> -->
                <td class="num">{swatch_html}{wl_nm:.2f}</td>
                <td class="num">{self.V:.2f}</td>
                <td class="num">{self.neff:.4f}</td>
                <td class="num">{S0:.2f}</td>
                <td class="num">({fmt_signed(S1)}, {fmt_signed(S2)}, {fmt_signed(S3)})</td>
            </tr>
            </tbody>
        </table>
        </div>
        """
        return html
    
