"""
fields.py

Defines the GuidedMode class, which represents a single eigenmode (ℓ, m, λ) of a step-index fiber.
Now includes a lightweight grid cache to avoid recomputing radial functions and phases when
sampling E/H on the *same* spatial grid repeatedly (e.g., for plotting, power integration,
animations along z, etc.).

Cache design (small & safe):
- Keyed by coordinate system ("xy" or "cyl") and a compact fingerprint of the grid
  (shape, extents, mean step, z if scalar/broadcastable). This avoids storing
  gigantic array hashes and keeps comparisons fast.
- Stored items: flattened rho/phi/z, shape, precomputed radial functions R0, Rp, Rm,
  and the longitudinal/azimuthal phase exp(i(kz z + ℓ φ)).
- Invalidation: automatic whenever A/B, kz, ℓ, wavelength, fibre, or V/b change,
  i.e., on new GuidedMode instance. You can also call clear_grid_cache().

Notes:
- The cache is *per-mode* (per GuidedMode instance). It never grows beyond a handful of grids
  you touch in a session. Use clear_grid_cache() if you need to free memory.
- If you pass a different grid (different shape or spacing), the cache just recomputes and stores it.

Author: Sebastian Golat (extended with grid caching)
"""

import numpy as np
from scipy.special import jv, kv, jvp, kve
from scipy.constants import epsilon_0 as eps0, mu_0 as mu0, c as c0
from .utils import units, _HAS_UNITS, _strip_unit
from .dispersion import b_to_neff, b_to_kz


# ---------------------------- small helpers ---------------------------------

def spin_to_cartesian(F0, Fp, Fm):
    Ex = (Fp + Fm) / np.sqrt(2)
    Ey = 1j * (Fp - Fm) / np.sqrt(2)
    Ez = F0
    return np.stack([Ex, Ey, Ez], axis=-1)


def cartesian_to_cylindrical(F_cart, phi):
    Fx = F_cart[..., 0]
    Fy = F_cart[..., 1]
    Fz = F_cart[..., 2]
    F_rho = Fx * np.cos(phi) + Fy * np.sin(phi)
    F_phi = -Fx * np.sin(phi) + Fy * np.cos(phi)
    return np.stack([F_rho, F_phi, Fz], axis=-1)


def cylindrical_to_cartesian(F_cyl, phi):
    F_rho = F_cyl[..., 0]
    F_phi = F_cyl[..., 1]
    Fz = F_cyl[..., 2]
    Fx = F_rho * np.cos(phi) - F_phi * np.sin(phi)
    Fy = F_rho * np.sin(phi) + F_phi * np.cos(phi)
    return np.stack([Fx, Fy, Fz], axis=-1)


def format_complex_auto(z, tol=1e-14):
    """Format complex/Quantity like Python default, but omit near-zero parts."""
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
    def __init__(self, fibre, ell, m, wavelength, mode_type=None, N_b=2000, *, a_plus=1.0, a_minus=0.0):
        self.fibre = fibre
        self.ell = ell
        self.m = m
        self.wavelength = wavelength
        self.mode_type = mode_type

        # Optional superposition coefficients for ℓ≠0 (degenerate ±|ℓ| pair)
        self.a_plus = complex(a_plus)
        self.a_minus = complex(a_minus)
        s0 = abs(self.a_plus)**2 + abs(self.a_minus)**2
        self.a_plus /= np.sqrt(s0)
        self.a_minus /= np.sqrt(s0)

        self.V = fibre.V(wavelength)
        self.b = fibre.b(ell, m, wavelength=self.wavelength, mode_type=mode_type, N_b=N_b)[0]
        if np.isnan(self.b) or not (0 < self.b < 1):
            raise ModeNotFoundError(
                f"Mode (ℓ={ell}, m={m}) does not exist at λ={wavelength}."
            )
        self.neff = b_to_neff(fibre, self.b, wavelength=self.wavelength)
        self.kz = b_to_kz(fibre, self.b, wavelength=self.wavelength)
        self.k0 = 2 * np.pi / _strip_unit(wavelength, units.m if _HAS_UNITS else None)

        self.A, self.B = self._compute_amplitudes()

        # Per-mode grid cache (see module docstring)
        # maps keys -> dict(rho_flat, phi_flat, z_flat, shape, R0, Rp, Rm, phi_phase, s_phase, z_phase)
        self._grid_cache = {}

    # ----------------------- material/geometry wrappers ----------------------
    def eps(self, rho):
        return self.fibre.eps(rho, self.wavelength)

    def mu(self, rho):
        return self.fibre.mu(rho, self.wavelength)

    def n(self, rho):
        return self.fibre.n(rho, self.wavelength)

    def k(self, rho):
        return self.k0 * self.n(rho)

    def kap(self, rho):
        k = self.k(rho)
        return np.sqrt(k**2 - self.kz**2 + 0j)

    # -------------------------- cache management -----------------------------
    def clear_grid_cache(self):
        """Drop all cached grids and precomputations for this mode."""
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
    def _power_normalisation(self, A, B):
        from .dispersion import _wDlnK
        fibre = self.fibre
        ell = int(abs(self.ell))
        V = self.V
        wl = self.wavelength
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

        kappasq1 = k1**2 - kz**2
        gammasq2 = kz**2 - k2**2

        if A == 0 and B == 0:
            return 1.0

        DlnJ = jvp(ell, u)/jv(ell, u)
        I1_plus = (u**2 - ell**2) / (2 * u**2) + DlnJ / u + DlnJ**2 / 2
        I1_minus = ell / (u**2)

        # DlnK = -(kve(ell - 1, w) + kve(ell+1, w))/(2*kve(ell, w))
        DlnK = _wDlnK(ell, w)/w
        I2_plus = (w**2 + ell**2) / (2 * w**2) - DlnK / w - DlnK**2 / 2
        I2_minus = -ell / (w**2)

        alpha1_plus = (eps1 * np.abs(A) ** 2 + mu1 * np.abs(B) ** 2) / (np.abs(A) ** 2 + np.abs(B) ** 2)
        alpha2_plus = (eps2 * np.abs(A) ** 2 + mu2 * np.abs(B) ** 2) / (np.abs(A) ** 2 + np.abs(B) ** 2)
        alpha_minus = (2 * np.imag(A * np.conj(B))) / (np.abs(A)**2 + np.abs(B)**2)

        term1 = (kz * k0) / kappasq1 * alpha1_plus * I1_plus
        term2 = (kz**2 + k1**2) / kappasq1 * alpha_minus * I1_minus
        term3 = (kz * k0) / gammasq2 * alpha2_plus * I2_plus
        term4 = (kz**2 + k2**2) / gammasq2 * alpha_minus * I2_minus

        sigma = c0 * np.pi * a**2 * (term1 + term2 + term3 + term4)

        # print(c0 * np.pi * a**2 * term1,c0 * np.pi * a**2 * term2,c0 * np.pi * a**2 * term3,c0 * np.pi * a**2 * term4)
        # # print(r"|κ₁|^2 =", kappasq1, r"|κ₂|^2 =", gammasq2)
        # print(r"α_+ =", alpha1_plus, r"α_2+ =", alpha2_plus, r"α_- =", alpha_minus)
        # print(r"I₁+ =", I1_plus, r"I₁- =", I1_minus, r"I₂+ =", I2_plus, r"I₂- =", I2_minus)
        # print("u =", u, r"w =", w)

        # --- Diagnostic and robust fallback ---
        # For hybrid modes (Im(A B*) != 0), numerical issues or sign mistakes in the
        # analytic form can lead to non‑physical σ (<=0) or large mismatch.
        # We estimate power numerically on a coarse polar grid and use it as a fallback.
        def _is_bad(x):
            return (not np.isfinite(x)) or (x <= 0)

        sigma_num = None
        use_numeric = _is_bad(sigma)
        if not use_numeric:
            # Compute numeric estimate only for hybrid cases or if sanity check fails
            ab_im = np.imag(np.conj(B) * A)
            if abs(ab_im) > 1e-12:
                sigma_num = self._estimate_power_numeric(A, B)
                # Compare relative mismatch if numeric is valid
                if np.isfinite(sigma_num) and sigma_num > 0:
                    rel = abs((sigma - sigma_num) / sigma_num)
                    # Allow small discrepancies (<5%); otherwise prefer numeric
                    if rel > 5e-2:
                        use_numeric = True
                else:
                    sigma_num = None
        import warnings
        if use_numeric:
            warnings.warn("Using numeric estimate for modal power", RuntimeWarning)
            print("Using numeric estimate for modal power")
            if sigma_num is None:
                sigma_num = self._estimate_power_numeric(A, B)
            if not np.isfinite(sigma_num) or sigma_num <= 0:
                raise RuntimeError("Normalization failed: analytic σ invalid and numeric estimate also invalid.")
            sigma = sigma_num

        N = 1.0 / np.sqrt(sigma)
        return N

    # def _estimate_power_numeric(self, A, B, r_max_factor=5.0, Nr=512, Ntheta=64, tol=5e-3, max_iter=6, decay_mult=12.0):
    #     """Estimate modal power via 1D radial integration with angle averaging.

    #     P ≈ ∫_0^{R} 2π r ⟨S_z(r,θ)⟩_θ dr, where S_z = 1/2 Re(E × H*)·ẑ.
    #     We use a modest number of angles and radial samples, and adaptively grow R
    #     until the integral converges within `tol` or `max_iter` is reached.

    #     Returns a float (SI units). On failure, returns NaN.
    #     """
    #     # Save original amplitudes and swap
    #     A0, B0 = self.A, self.B
    #     try:
    #         self.A, self.B = complex(A), complex(B)
    #         a = float(self.fibre.core_radius)
    #         # Estimate cladding decay length 1/gamma (guided ⇒ gamma>0)
    #         try:
    #             n2 = float(self.fibre.n_clad(self.wavelength))
    #             k2 = self.k0 * n2
    #             gam2 = float(self.kz**2 - k2**2)
    #             gamma = np.sqrt(gam2) if gam2 > 0 else 0.0
    #         except Exception:
    #             gamma = 0.0
    #         r_dec = (1.0 / gamma) if (isinstance(gamma, (int, float)) and gamma > 0) else 0.0
    #         base = max(r_max_factor * a, a + decay_mult * r_dec)
    #         r_max = base if base > 0 else r_max_factor * max(a, 1.0)
    #         P_prev = None
    #         best = None

    #         # Precompute angular samples
    #         thetas = np.linspace(0.0, 2*np.pi, int(max(16, Ntheta)), endpoint=False)

    #         for _ in range(int(max_iter)):
    #             # Radial samples
    #             Nr_i = int(max(128, Nr))
    #             rho = np.linspace(0.0, r_max, Nr_i)
    #             # Build mesh for angle averaging: (Ntheta, Nr)
    #             Rho = np.tile(rho, (len(thetas), 1))
    #             Phi = np.tile(thetas[:, None], (1, Nr_i))

    #             # Compute fields and S_z
    #             E = self.E(rho=Rho, phi=Phi, z=0)
    #             H = self.H(rho=Rho, phi=Phi, z=0)
    #             S = 0.5 * np.real(np.cross(E, np.conj(H)))
    #             Sz = S[..., 2]  # shape (Ntheta, Nr)
    #             # Angle-average at each radius
    #             Sz_avg = np.mean(Sz, axis=0)  # (Nr,)
    #             # Integrate 2π ∫ r Sz_avg(r) dr
    #             P = 2*np.pi * np.trapz(rho * Sz_avg, rho)
    #             P = float(np.real(P))
    #             best = P

    #             if P_prev is not None:
    #                 rel = abs((P - P_prev) / (P if P != 0 else 1.0))
    #                 if rel < tol and P > 0:
    #                     break
    #             P_prev = P
    #             r_max *= 2.0

    #         return best if (best is not None) else float('nan')
    #     except Exception:
    #         return float('nan')
    #     finally:
    #         self.A, self.B = A0, B0
    def _estimate_power_numeric(self, A, B, r_max_factor=5.0, Nr=512, Ntheta=64,
                            tol=5e-3, max_iter=6, decay_mult=12.0):
        """Estimate modal power via 1D radial integration with angle averaging."""
        # Guarded save
        had_AB = hasattr(self, "A") and hasattr(self, "B")
        if had_AB:
            A0, B0 = self.A, self.B

        try:
            # Use the trial amplitudes locally
            self.A, self.B = complex(A), complex(B)

            a = float(self.fibre.core_radius)
            # Estimate cladding decay length 1/gamma (guided ⇒ gamma>0)
            try:
                n2 = float(self.fibre.n_clad(self.wavelength))
                k2 = self.k0 * n2
                gam2 = float(self.kz**2 - k2**2)
                gamma = np.sqrt(gam2) if gam2 > 0 else 0.0
            except Exception:
                gamma = 0.0

            r_dec = (1.0 / gamma) if (isinstance(gamma, (int, float)) and gamma > 0) else 0.0
            base = max(r_max_factor * a, a + decay_mult * r_dec)
            r_max = base if base > 0 else r_max_factor * max(a, 1.0)

            P_prev = None
            best = None

            thetas = np.linspace(0.0, 2*np.pi, int(max(16, Ntheta)), endpoint=False)

            for _ in range(int(max_iter)):
                Nr_i = int(max(128, Nr))
                rho = np.linspace(0.0, r_max, Nr_i)
                Rho = np.tile(rho, (len(thetas), 1))
                Phi = np.tile(thetas[:, None], (1, Nr_i))

                E = self.E(rho=Rho, phi=Phi, z=0)  # uses the temporary self.A/B set above
                H = self.H(rho=Rho, phi=Phi, z=0)

                S = 0.5 * np.real(np.cross(E, np.conj(H)))
                Sz_avg = np.mean(S[..., 2], axis=0)
                P = 2*np.pi * np.trapz(rho * Sz_avg, rho)
                P = float(np.real(P))
                best = P

                if P_prev is not None:
                    denom = P if P != 0 else 1.0
                    if abs((P - P_prev) / denom) < tol and P > 0:
                        break
                P_prev = P
                r_max *= 2.0

            return best if (best is not None) else float('nan')

        except Exception:
            return float('nan')

        finally:
            # Restore only if we had them
            if had_AB:
                self.A, self.B = A0, B0
            else:
                # Clean up the temporaries to avoid leaving half-initialised attributes
                if hasattr(self, "A"): del self.A
                if hasattr(self, "B"): del self.B

    def _compute_amplitudes(self, Normalise=False):
        fibre = self.fibre
        ell = self.ell
        b = self.b
        V = self.V
        wl = self.wavelength
        ne = self.neff

        eps1 = fibre._eval(fibre.eps_core, wl)
        eps2 = fibre._eval(fibre.eps_clad, wl)
        mu1 = fibre._eval(fibre.mu_core, wl)
        mu2 = fibre._eval(fibre.mu_clad, wl)

        u = V * np.sqrt(1 - b)
        w = V * np.sqrt(b)

        DlnJ = (jvp(ell, u) / jv(ell, u))
        DlnK = -(kve(ell - 1, w) + kve(ell+1, w))/(2*kve(ell, w))

        phi_eps = ((u * w / V) ** 2) * (eps1 / u * DlnJ + eps2 / w * DlnK)
        phi_mu  = ((u * w / V) ** 2) * (mu1 / u * DlnJ + mu2 / w * DlnK)

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

        N = self._power_normalisation(A, B) if Normalise else 1.0
        return N*A, N*B

    # --------------------------- radial functions ----------------------------
    def _radial_function(self, s, rho):
        rho0 = self.fibre.core_radius

        k1 = self.k0 * self.fibre.n_core(self.wavelength)
        k2 = self.k0 * self.fibre.n_clad(self.wavelength)
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

    def _spin_components(self, field, rho_flat, phi_flat, z_flat, phi_phase, s_phase, z_phase):
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

    # ------------------------------- public API ------------------------------
    def E(self, rho=None, phi=None, z=0, *, x=None, y=None, Normalise=False):
        (rho_f, phi_f, z_f, shape, phi_phase, s_phase, z_phase, R0, Rp, Rm) = self._prepare_grid(
            rho=rho, phi=phi, z=z, x=x, y=y
        )
        F0, c_plus, c_minus, phi_phase, s_phase, z_phase = self._spin_components('E', rho_f, phi_f, z_f, phi_phase, s_phase, z_phase)
        F0_plus = F0 * R0 * phi_phase * z_phase
        Fp_plus = c_plus * Rp * np.conj(s_phase) * phi_phase * z_phase
        Fm_plus = c_minus * Rm * s_phase * phi_phase * z_phase
        F0_minus = F0 * R0 * np.conj(phi_phase) * z_phase
        Fp_minus = c_minus * Rm * np.conj(s_phase) * np.conj(phi_phase) * z_phase
        Fm_minus = c_plus * Rp * s_phase * np.conj(phi_phase) * z_phase
        F0 = self.a_plus * F0_plus + self.a_minus * F0_minus
        Fp = self.a_plus * Fp_plus + self.a_minus * Fp_minus
        Fm = self.a_plus * Fm_plus + self.a_minus * Fm_minus

        return spin_to_cartesian(F0, Fp, Fm).reshape(*shape, 3)

    def H(self, rho=None, phi=None, z=0, *, x=None, y=None, Normalise=False):
        (rho_f, phi_f, z_f, shape, phi_phase, s_phase, z_phase, R0, Rp, Rm) = self._prepare_grid(
            rho=rho, phi=phi, z=z, x=x, y=y
        )
        F0, c_plus, c_minus, phi_phase, s_phase, z_phase = self._spin_components('H', rho_f, phi_f, z_f, phi_phase, s_phase, z_phase)
        F0_plus = F0 * R0 * phi_phase * z_phase
        Fp_plus = c_plus * Rp * np.conj(s_phase) * phi_phase * z_phase
        Fm_plus = c_minus * Rm * s_phase * phi_phase * z_phase
        F0_minus = -F0 * R0 * np.conj(phi_phase) * z_phase
        Fp_minus = -c_minus * Rm * np.conj(s_phase) * np.conj(phi_phase) * z_phase
        Fm_minus = -c_plus * Rp * s_phase * np.conj(phi_phase) * z_phase
        F0 = self.a_plus * F0_plus + self.a_minus * F0_minus
        Fp = self.a_plus * Fp_plus + self.a_minus * Fp_minus
        Fm = self.a_plus * Fm_plus + self.a_minus * Fm_minus

        return spin_to_cartesian(F0, Fp, Fm).reshape(*shape, 3)

    def Power(self, rho=None, phi=None, z=0, *, x=None, y=None):
        if x is not None and y is not None:
            x = np.asarray(x); y = np.asarray(y)
            S = 0.5 * np.real(np.cross(self.E(x=x, y=y, z=z), np.conj(self.H(x=x, y=y, z=z))))
            dx = np.mean(np.diff(x[0, :])) if x.ndim == 2 else np.mean(np.diff(x))
            dy = np.mean(np.diff(y[:, 0])) if y.ndim == 2 else np.mean(np.diff(y))
            dA = dx * dy
            P = np.sum(S[..., 2]) * dA
            return P
        elif rho is not None and phi is not None:
            rho = np.asarray(rho); phi = np.asarray(phi)
            S = 0.5 * np.real(np.cross(self.E(rho=rho, phi=phi, z=z), np.conj(self.H(rho=rho, phi=phi, z=z))))
            # crude average area weight in polar sampling (assumes near-uniform grids)
            # drho = np.mean(np.diff(rho[0, :])) if rho.ndim == 2 else np.mean(np.diff(rho))
            # dphi = np.mean(np.diff(phi[:, 0])) if phi.ndim == 2 else np.mean(np.diff(phi))
            drho = rho[0, 1] - rho[0, 0]
            dphi = phi[1, 0] - phi[0, 0]
            dA = (rho * drho * dphi)
            P = np.sum(S[..., 2] * dA)
            return P
        else:
            raise ValueError("Provide either (x, y) or (rho, phi) as input.")

    def __repr__(self):
        return (
            f"<GuidedMode ℓ={self.ell}, m={self.m}, "
            f"λ={self.wavelength:.2e}, V={self.V:.2e}, neff={self.neff:.6f}, "
            # f"A={self.A:.3f}, B={self.B:.3f}>"
        )

    # ------------------------------ labelling --------------------------------
    def _mode_kind_hybrid(self, atol=1e-9):
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

    def mode_label(self):
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
            n = self.m
            return f"{mt}<sub>{self.ell}{n}</sub>"

    # ------------------------------ rich display -----------------------------
    def _repr_html_(self):
        from .utils import _HAS_UNITS, wavelength_to_rgb, wavelength_band_label_nm
        try:
            wl_nm = self.wavelength.to_value('nm') if _HAS_UNITS else float(self.wavelength) * 1e9
        except Exception:
            wl_nm = None

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

        # stokes parameters for ell\neq 0
        if self.ell != 0:
            S1 = 2 * np.real(self.a_plus * np.conj(self.a_minus))
            S2 = 2 * np.imag(self.a_plus * np.conj(self.a_minus))
            S3 = np.abs(self.a_plus) ** 2 - np.abs(self.a_minus) ** 2
        elif self.mode_type == "TE":
            S1, S2, S3 = -1.0, 0.0, 0.0
        elif self.mode_type == "TM":
            S1, S2, S3 = 1.0, 0.0, 0.0

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
                <!--<th scope="col" style="text-align:center;">ℓ</th>
                <th scope="col" style="text-align:center;">m</th> -->
                <th scope="col" class="num">λ [nm]</th>
                <th scope="col" class="num">V</th>
                <th scope="col" class="num">n<sub>e</sub></th>
                <th scope="col" class="num">s<sub>1</sub></th>
                <th scope="col" class="num">s<sub>2</sub></th>
                <th scope="col" class="num">s<sub>3</sub></th>
            </tr>
            </thead>
            <tbody>
            <tr>
                <td style="text-align:center;">{self.mode_label()}</td>
                <!-- <td style="text-align:center;">{self.ell}</td>
                <td style="text-align:center;">{self.m}</td> -->
                <td class="num">{swatch_html}{wl_nm:.2f}</td>
                <td class="num">{self.V:.2f}</td>
                <td class="num">{self.neff:.4f}</td>
                <td class="num">{S1:.2f}</td>
                <td class="num">{S2:.2f}</td>
                <td class="num">{S3:.2f}</td>
            </tr>
            </tbody>
        </table>
        </div>
        """
        return html
    @staticmethod
    def _repr_html_multi(modes):
        """
        Return ONE HTML table for an iterable of GuidedMode objects,
        using the same styling/columns as the single-row _repr_html_.
        """
        from .utils import _HAS_UNITS, wavelength_to_rgb, wavelength_band_label_nm

        modes = [m for m in modes if m is not None]
        if not modes:
            return ""

        # --- same swatch helper as in your single renderer ---
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

        rows = []
        for m in modes:
            # Wavelength + swatch
            try:
                wl_nm = m.wavelength.to_value('nm') if _HAS_UNITS else float(m.wavelength) * 1e9
                wl_str = f"{wl_nm:.2f}"
            except Exception:
                wl_nm, wl_str = None, f"{m.wavelength:.2e}"

            label, gray = wavelength_band_label_nm(wl_nm) if wl_nm is not None else (None, None)
            if wl_nm is not None and 380 <= wl_nm <= 780:
                R, G, B = wavelength_to_rgb(wl_nm)
                swatch_html = _swatch(f"rgb({int(R*255)}, {int(G*255)}, {int(B*255)})")
            elif label is not None:
                swatch_html = _swatch(gray, label)
            else:
                swatch_html = ""

            # Stokes
            if getattr(m, "ell", 0) != 0 and hasattr(m, "a_plus") and hasattr(m, "a_minus"):
                S1 = 2 * np.real(m.a_plus * np.conj(m.a_minus))
                S2 = 2 * np.imag(m.a_plus * np.conj(m.a_minus))
                S3 = np.abs(m.a_plus)**2 - np.abs(m.a_minus)**2
            elif getattr(m, "ell", 0) == 0 and getattr(m, "mode_type", None) == "TE":
                S1, S2, S3 = -1.0, 0.0, 0.0
            else:
                S1, S2, S3 = 1.0, 0.0, 0.0

            rows.append(f"""
            <tr>
                <td style="text-align:center;">{m.mode_label()}</td>
                <td class="num">{swatch_html}{wl_str}</td>
                <td class="num">{m.V:.2f}</td>
                <td class="num">{m.neff:.4f}</td>
                <td class="num">{S1:.2f}</td>
                <td class="num">{S2:.2f}</td>
                <td class="num">{S3:.2f}</td>
            </tr>
            """)

        # one header, many rows
        html = f"""
        <div class="anafibre-guidedmode" aria-label="Guided mode summary"
            style="color-scheme: light dark; --bg: Canvas; --fg: CanvasText;
                    --border: color-mix(in srgb, var(--fg) 18%, var(--bg));
                    --header: color-mix(in srgb, var(--fg) 8%, var(--bg));
                    --row: color-mix(in srgb, var(--fg) 4%, var(--bg));
                    font-variant-numeric: tabular-nums;">
        <style>
        .anafibre-guidedmode table {{
            width: auto; background: var(--bg); color: var(--fg);
            border: 1px solid var(--border); border-radius: 10px;
            border-collapse: separate; border-spacing: 0; overflow: hidden;
            box-shadow: 0 2px 10px color-mix(in srgb, var(--fg) 10%, transparent);
        }}
        .anafibre-guidedmode th, .anafibre-guidedmode td {{
            padding: .55rem .5rem; vertical-align: middle;
        }}
        .anafibre-guidedmode thead th {{
            background: var(--header); font-weight: 600;
        }}
        .anafibre-guidedmode td.num, .anafibre-guidedmode th.num {{ text-align: center; }}
        .anafibre-guidedmode tbody tr:nth-child(odd) td {{ background: var(--row); }}
        </style>

        <table>
        <thead>
            <tr>
            <th scope="col" style="text-align:center;">Mode</th>
            <th scope="col" class="num">λ [nm]</th>
            <th scope="col" class="num">V</th>
            <th scope="col" class="num">n<sub>e</sub></th>
            <th scope="col" class="num">s<sub>1</sub></th>
            <th scope="col" class="num">s<sub>2</sub></th>
            <th scope="col" class="num">s<sub>3</sub></th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
        </table>
        </div>
        """
        return html
