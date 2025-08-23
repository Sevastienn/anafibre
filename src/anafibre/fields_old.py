"""
fields.py

Defines the GuidedMode class, which represents a single eigenmode (ℓ, m, λ) of a step-index fiber.
This version supports evaluation of field parameters but not superpositions yet.
"""

import numpy as np
from scipy.special import jv, kv, jvp, kvp, kve
from scipy.constants import epsilon_0 as eps0, mu_0 as mu0, c as c0
from .utils import units, _HAS_UNITS, _strip_unit
from .dispersion import b_to_neff, b_to_kz

def spin_to_cartesian(F0, Fp, Fm):
    Ex = (Fp + Fm) / np.sqrt(2)
    Ey = 1j * (Fp - Fm) / np.sqrt(2)
    Ez = F0
    return np.stack([Ex, Ey, Ez], axis=-1)

def cartesian_to_cylindrical(F_cart, phi):
    """
    F_cart: (..., 3) array with [F_x, F_y, F_z] in last axis
    phi: array matching field shape (broadcastable)
    Returns (..., 3) array with [F_rho, F_phi, F_z]
    """
    Fx = F_cart[..., 0]
    Fy = F_cart[..., 1]
    Fz = F_cart[..., 2]
    F_rho =  Fx * np.cos(phi) + Fy * np.sin(phi)
    F_phi = -Fx * np.sin(phi) + Fy * np.cos(phi)
    return np.stack([F_rho, F_phi, Fz], axis=-1)

def cylindrical_to_cartesian(F_cyl, phi):
    """
    F_cyl: (..., 3) array with [F_rho, F_phi, F_z] in last axis
    phi: array matching field shape (broadcastable)
    Returns (..., 3) array with [F_x, F_y, F_z]
    """
    F_rho = F_cyl[..., 0]
    F_phi = F_cyl[..., 1]
    Fz    = F_cyl[..., 2]
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


class GuidedMode:
    def __init__(self, fibre, ell, m, wavelength, mode_type=None, N_b=2000):
        self.fibre = fibre
        self.ell = ell
        self.m = m
        self.wavelength = wavelength
        self.mode_type = mode_type
        

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
        # print(wavelength, self.wavelength, self.neff, self.b, self.kz, self.V, self.A, self.B)

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

    # def _power_normalisation_numeric(self, A, B):

    def _power_normalisation(self, A, B):

        fibre = self.fibre
        ell = self.ell
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
            return 1.0  # or raise error

        # Core region integrals (as before)
        DlnJ = jvp(ell, u)/jv(ell, u)
        I1_plus = (u**2 - ell**2) / (2 * u**2) + DlnJ / u + DlnJ**2 / 2 
        I1_minus = ell / (u**2)

        # Cladding region integrals
        # DlnK = kvp(ell, w)/kv(ell, w)
        DlnK = -(kve(ell - 1, w) + kve(ell+1, w))/(2*kve(ell, w))
        I2_plus = (w**2 + ell**2) / (2 * w**2) - DlnK / w - DlnK**2 / 2
        I2_minus = -ell / (w**2)

        # print(f"I₊1={I1_plus:.3e}, I₊2={I2_plus:.3e}, I₋1={I1_minus:.3e}, I₋2={I2_minus:.3e}")


        # α coefficients for core (1) and cladding (2)
        alpha1_plus = (eps1 * np.abs(A) ** 2 + mu1 * np.abs(B) ** 2) / (np.abs(A) ** 2 + np.abs(B) ** 2)
        alpha2_plus = (eps2 * np.abs(A) ** 2 + mu2 * np.abs(B) ** 2) / (np.abs(A) ** 2 + np.abs(B) ** 2)
        alpha_minus = 2 * np.imag(np.conj(B) * A) / (np.abs(A) ** 2 + np.abs(B) ** 2)
        # print(f"α₋={alpha_minus:.3e}, α₊1={alpha1_plus:.3e}, α₊2={alpha2_plus:.3e}")

        # Compute the normalization constant σ_lm
        term1 = (kz * k0) / kappasq1 * alpha1_plus * I1_plus
        term2 = (kz**2 + k1**2) / kappasq1 * alpha_minus * I1_minus
        term3 = (kz * k0) / gammasq2 * alpha2_plus * I2_plus
        term4 = (kz**2 + k2**2) / gammasq2 * alpha_minus * I2_minus
        print(f"Normalization terms: {term1:.3e}, {term2:.3e}, {term3:.3e}, {term4:.3e}")

        # Total power normalization constant
        sigma = c0 * np.pi * a**2 * (term1 + term2 + term3 + term4)
        print(f"σ_lm = {sigma:.3e} (A={A:.3e}, B={B:.3e})")
        
        if sigma == 0 or np.isnan(sigma) or np.isinf(sigma):
            raise RuntimeError("Normalization failed: σ_lm = 0 or invalid.")

        N = 1 / np.sqrt(sigma)
        return N


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

        # Jl = jv(ell, u)
        # Jlp = jvp(ell, u)
        # Kl = kv(ell, w)
        # Klp = kvp(ell, w)
        DlnJ = (jvp(ell, u) / jv(ell, u))
        # DlnK = (kvp(ell, w) / kv(ell, w))
        DlnK = -(kve(ell - 1, w) + kve(ell+1, w))/(2*kve(ell, w))

        phi_eps = ((u * w / V) ** 2) * (eps1 / u * DlnJ + eps2 / w * DlnK)
        phi_mu  = ((u * w / V) ** 2) * (mu1 / u * DlnJ + mu2 / w * DlnK)

        if ell == 0:
            mt = self.mode_type.lower() if self.mode_type is not None else None
            if mt is None:
                is_te = np.isclose(phi_mu, 0, atol=1e-9)
                is_tm = np.isclose(phi_eps, 0, atol=1e-9)
                if is_te and not is_tm:
                    mt = "te"
                    phi_mu = 0*phi_eps  # TE mode
                elif is_tm and not is_te:
                    mt = "tm"
                    phi_eps = 0*phi_mu  # TM mode
            if mt == "te":
                A, B = 0.0, 1.0j
                # print("Φ_μ =", phi_mu, "Φ_ε =", phi_eps)
            elif mt == "tm":
                A, B = 1.0, 0.0
                # print("Φ_μ =", phi_mu, "Φ_ε =", phi_eps)
            else:
                raise ValueError("mode_type must be 'TE', 'TM', or None for ℓ = 0.")

        else: #ℓ ≠ 0: hybrid mode
            nu_eps = 1j * phi_eps / (ell * ne)
            # nu_mu = 1j * (ell * ne) / phi_mu
            nu = np.sqrt(-phi_eps / phi_mu+0j)
            # Match sign to nu_eps
            if np.real(nu * np.conj(nu_eps)) < 0:
                nu = -nu
            A = 1 / np.sqrt(1 + np.abs(nu)**2)
            B = nu * A
            # print(f"ν_eps = {nu_eps:.3e}, ν_mu = {nu_mu:.3e}, ν = {nu:.3e}, Δνe = {nu_eps - nu:.3e}, Δνm = {nu_mu - nu:.3e}, Δν = {nu_eps - nu_mu:.3e}")
        
        if Normalise:
            N = self._power_normalisation(A,B)
        else:
            N = 1.0
        # if Normalise:
        #     rho = np.linspace(0, fibre.core_radius, 1000)
        #     phi = np.linspace(0, 2 * np.pi, 1000)
        #     Power = self.Power(rho, phi)
        #     N = 1 / np.sqrt(Power)
        # else:
        #     N = 1.0

        return N*A, N*B

    def _radial_function(self, s, rho):
        rho0 = self.fibre.core_radius
        # kap = self.kap(rho)

        k1 = self.k0 * self.fibre.n_core(self.wavelength)
        k2 = self.k0 * self.fibre.n_clad(self.wavelength)
        kz = self.kz

        #  # Region-specific real arguments
        if np.iscomplex(kz):
            kap1 = np.sqrt(k1**2 - kz**2 + 0j)
            gam2 = np.sqrt(kz**2 - k2**2 + 0j)
        else:
            kap1 = np.sqrt(k1**2 - kz**2)
            gam2 = np.sqrt(kz**2 - k2**2)
            

        inside = rho < rho0
        R = np.empty_like(rho, dtype=complex)
        # R[inside] = jv(self.ell - s, kap[inside] * rho[inside]) / jv(self.ell, kap[inside] * rho0)
        # R[~inside] = (1j)**(s)*kv(self.ell - s, -1j*kap[~inside] * rho[~inside]) / kv(self.ell, -1j*kap[~inside] * rho0)

        R[inside] = jv(self.ell - s, kap1 * rho[inside]) / jv(self.ell, kap1 * rho0)
        R[~inside] = (1j)**(s)*kv(self.ell - s, gam2 * rho[~inside]) / kv(self.ell, gam2 * rho0)
        return R

    def _spin_components(self, field, rho_flat, phi_flat, z_flat):
        A, B = self.A, self.B
        kz = self.kz
        k0 = self.k0
        k1 = self.k0 * self.fibre.n_core(self.wavelength)
        ell = self.ell

        R0 = self._radial_function(0, rho_flat)
        Rp = self._radial_function(+1, rho_flat)
        Rm = self._radial_function(-1, rho_flat)
        kap = self.kap(rho_flat)

        total_phase = np.exp(1j * (kz * z_flat + ell * phi_flat))

        if field.lower() == 'e':
            A1, A2 = A, B
            alpha = self.mu(rho_flat)
            alpha1 = self.fibre._eval(self.fibre.mu_core, self.wavelength)
            beta0 = eps0
        elif field.lower() == 'h':
            A1, A2 = B, -A
            alpha = self.eps(rho_flat)
            alpha1 = self.fibre._eval(self.fibre.eps_core, self.wavelength)
            beta0 = mu0
        else:
            raise ValueError("field must be 'E' or 'H'")
        
        # Compute the field components
        F0 = (A1/np.sqrt(beta0)) * R0 * total_phase
        Fp = ((+1j * kz * A1 - k0 * alpha * A2) / (kap * np.sqrt(2*beta0))) * Rp * np.exp(-1j * phi_flat) * total_phase
        Fm = ((-1j * kz * A1 - k0 * alpha * A2) / (kap * np.sqrt(2*beta0))) * Rm * np.exp(+1j * phi_flat) * total_phase
        # kap1 = np.sqrt(k1**2 - kz**2)
        # print(f"Fp: {(+1j * kz * A1 - k0 * alpha1 * A2) / (kap1 * np.sqrt(2))}, Fm: {(-1j * kz * A1 - k0 * alpha1 * A2) / (kap1 * np.sqrt(2))}")

        return F0, Fp, Fm
    
    
    def E(self, rho=None, phi=None, z=0, *, x=None, y=None, Normalise=False):
        if x is not None and y is not None:
            x = np.asarray(x)
            y = np.asarray(y)
            rho = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
        if rho is None or phi is None:
            raise ValueError("Provide either (rho, phi) or (x, y) as input.")

        rho = np.asarray(rho)
        phi = np.asarray(phi)
        z = np.asarray(z)
        rho, phi, z = np.broadcast_arrays(rho, phi, z)
        shape = rho.shape
        rho_flat = rho.ravel()
        phi_flat = phi.ravel()
        z_flat = z.ravel()
        # if Normalise:
        #     if x is not None and y is not None:
        #         N = 1 / np.sqrt(self.Power(x=x, y=y, z=z))
        #     elif rho is not None and phi is not None:
        #         N = 1 / np.sqrt(self.Power(rho=rho, phi=phi, z=z))
        #     else:
        #         N = 1.0
        # else:
        #     N = 1.0

        F0, Fp, Fm = self._spin_components('E', rho_flat, phi_flat, z_flat)
        return spin_to_cartesian(F0, Fp, Fm).reshape(*shape, 3)

    def H(self, rho=None, phi=None, z=0, *, x=None, y=None, Normalise=False):
        if x is not None and y is not None:
            x = np.asarray(x)
            y = np.asarray(y)
            rho = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
        if rho is None or phi is None:
            raise ValueError("Provide either (rho, phi) or (x, y) as input.")

        rho = np.asarray(rho)
        phi = np.asarray(phi)
        z = np.asarray(z)
        rho, phi, z = np.broadcast_arrays(rho, phi, z)
        shape = rho.shape
        rho_flat = rho.ravel()
        phi_flat = phi.ravel()
        z_flat = z.ravel()
        # if Normalise:
        #     if x is not None and y is not None:
        #         N = 1 / np.sqrt(self.Power(x=x, y=y, z=z))
        #     elif rho is not None and phi is not None:
        #         N = 1 / np.sqrt(self.Power(rho=rho, phi=phi, z=z))
        #     else:
        #         N = 1.0
        # else:
        #     N = 1.0

        F0, Fp, Fm = self._spin_components('H', rho_flat, phi_flat, z_flat)
        return spin_to_cartesian(F0, Fp, Fm).reshape(*shape, 3)

    def Power(self, rho=None, phi=None, z=0, *, x=None, y=None):
        if x is not None and y is not None:
            x = np.asarray(x)
            y = np.asarray(y)
            z = np.asarray(z)
            S = 0.5 * np.real(np.cross(self.E(x=x, y=y, z=z), np.conj(self.H(x=x, y=y, z=z))))
            dx = np.mean(np.diff(x[0, :]))
            dy = np.mean(np.diff(y[:, 0]))
            dA = dx * dy
            P = np.sum(S[..., 2]) * dA
            return P
        elif rho is not None and phi is not None:
            rho = np.asarray(rho)
            phi = np.asarray(phi)
            z = np.asarray(z)
            S = 0.5 * np.real(np.cross(self.E(rho=rho, phi=phi, z=z), np.conj(self.H(rho=rho, phi=phi, z=z))))
            drho = np.mean(np.diff(rho[0, :]))
            dphi = np.mean(np.diff(phi[:, 0]))
            dA = (rho * drho * dphi).mean()
            P = np.sum(S[..., 2]) * dA
            return P
        else:
            raise ValueError("Provide either (x, y) or (rho, phi) as input.")

    def __repr__(self):
        return (
            f"<GuidedMode ℓ={self.ell}, m={self.m}, "
            f"λ={self.wavelength:.2e}, V={self.V:.2e}, neff={self.neff:.6f}, "
            f"A={self.A:.3f}, B={self.B:.3f}>"
        )
   
    def _mode_kind_hybrid(self, atol=1e-9):
        """
        Decide HE vs EH for ℓ≠0.
        Primary rule (IEEE): |Ez| > η0|Hz| ⇔ |A| > |B| -> HE; else EH.
        Tie-break: if |A|≈|B|, use sign of Im(A B*): + → HE, − → EH.
        """
        A, B = self.A, self.B

        # Tie-breaker using requested criterion
        s = np.sign(self.ell)*np.imag(A * np.conj(B))
        if s > 0:
            return "HE"
        elif s < 0:
            return "EH"
        else:
            # Perfect tie: fall back to m parity (dielectric heuristic)
            return "HE" if (self.m % 2 == 1) else "EH"
    def _radial_n(self):
        """Map root index m to conventional radial index n."""
        if self.ell == 0:
            mt = (self.mode_type or "").upper()
            if mt not in {"TE", "TM"}:
                return (self.m + 1) // 2  # == ceil(m/2)
            else:
                return self.m
        return (self.m + 1) // 2  # == ceil(m/2)

    def mode_label(self):
        """
        Human-friendly mode label:
        - ℓ = 0 → TE₀m or TM₀m (from self.mode_type or inferred)
        - ℓ > 0 → HEℓm / EHℓm (rule above)
        Returns plain HTML with subscripts.
        """
        n = self._radial_n()
        if self.ell == 0:
            mt = (self.mode_type or "").upper()
            if mt not in {"TE", "TM"}:
                # If not provided, infer from A/B exactly as you already do
                mt = "TE" if abs(self.A) < 1e-15 else ("TM" if abs(self.B) < 1e-15 else "TE")
            return f"{mt}<sub>0{n}</sub>"

        kind = self._mode_kind_hybrid()
        return f"{kind}<sub>{self.ell}{n}</sub>"
    def _repr_html_(self):
        # Local import to avoid polluting module namespace
        from .utils import _HAS_UNITS, wavelength_to_rgb, wavelength_band_label_nm

        # Format wavelength (nm) and build a small color swatch
        try:
            wl_nm = self.wavelength.to_value('nm') if _HAS_UNITS else float(self.wavelength) * 1e9
            wl_str = f"{wl_nm:.2f}"
        except Exception:
            wl_nm = None
            wl_str = f"{self.wavelength:.2e}"

        # if wl_nm is not None and 380 <= wl_nm <= 780:
        #     R, G, B = wavelength_to_rgb(wl_nm)  # returns floats 0..1
        #     swatch_style = f"background: rgb({int(R*255)}, {int(G*255)}, {int(B*255)});"
        #     swatch_html = f'<span style="display:inline-block;width:0.8em;height:0.8em;border:1px solid rgba(0,0,0,.25);border-radius:2px;margin-right:.4em;vertical-align:middle;{swatch_style}"></span>'
        # else:
        #     swatch_html = ""
        # ... after wl_nm / wl_str are set:
        # label, gray = wavelength_band_label_nm(wl_nm) if wl_nm is not None else (None, None)

        # if wl_nm is not None and 380 <= wl_nm <= 780:
        #     R, G, B = wavelength_to_rgb(wl_nm)  # floats 0..1
        #     swatch_style = f"background: rgb({int(R*255)}, {int(G*255)}, {int(B*255)});"
        #     swatch_html = (
        #         '<span style="display:inline-block;width:0.8em;height:0.8em;'
        #         'border:1px solid rgba(0,0,0,.25);border-radius:2px;'
        #         'margin-right:.4em;vertical-align:middle;'
        #         f'{swatch_style}"></span>'
        #     )
        # elif label is not None:
        #     swatch_html = (
        #         f'<span title="{label}" style="display:inline-block;'
        #         'min-width:1.8em;height:0.9em;line-height:0.9em;'
        #         'text-align:center;font-size:0.7em;font-weight:600;'
        #         'color:white;border-radius:2px;margin-right:.4em;'
        #         'vertical-align:middle;'
        #         f'background:{gray};">{label}</span>'
        #     )
        # else:
        #     swatch_html = ""
        # from .utils import wavelength_to_rgb, wavelength_band_label_nm

        # ...

        def _swatch(bg_css, label=None):
            """Small rounded square swatch; optional tiny label inside with dynamic font size."""
            text = (label or "")
            title = f' title="{label}"' if label else ""
            return (
                f'<span{title} style="display:inline-flex;align-items:center;justify-content:center;'
                'width:1.5em;height:1.5em;margin-right:.4em;margin-bottom:0.3em;'
                'border:1px solid rgba(0,0,0,.25);border-radius:2px;'
                f'background:{bg_css};color:white;'
                # dynamic font sizing: shrinks if label is long, grows if short
                'font-size:clamp(0.45em, 0.55em, 0.65em);'
                'font-weight:700;letter-spacing:.02em;vertical-align:middle;'
                'overflow:hidden;white-space:nowrap;text-overflow:ellipsis;">'
                f'{text}'
                '</span>'
            )

        # after wl_nm / wl_str are set:
        label, gray = wavelength_band_label_nm(wl_nm) if wl_nm is not None else (None, None)

        if wl_nm is not None and 380 <= wl_nm <= 780:
            R, G, B = wavelength_to_rgb(wl_nm)  # floats 0..1
            swatch_html = _swatch(f"rgb({int(R*255)}, {int(G*255)}, {int(B*255)})")
        elif label is not None:
            swatch_html = _swatch(gray, label)
        else:
            swatch_html = ""
        
        # make format_complex_auto(B/A) if A≠0 otherwise print i∞
        def format_nu_html(A, B, tol=1e-15):
            """
            Returns HTML for ν = B/A.
            - If |A|>tol → prints finite ν via your format_complex_auto.
            - If |A|≤tol → prints '± i∞' with a stable sign chosen from Φ's.
            """
            # Finite case
            if abs(A) > tol:
                return format_complex_auto(1j* B / A)

            return f"− &infin;"
        nu_html = format_nu_html(self.A, self.B)

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
            /* subtle, theme-aware shadow */
            box-shadow: 0 2px 10px color-mix(in srgb, var(--fg) 10%, transparent);
            }}
            .anafibre-guidedmode caption {{
            padding: .65rem .9rem;
            font-weight: 600;
            text-align: center;
            background: var(--header);
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
        <!-- <caption>GuidedMode</caption> -->
            <thead>
            <tr>
                <th scope="col" style="text-align:center;">Mode</th>
                <th scope="col" style="text-align:center;">ℓ</th>
                <th scope="col" style="text-align:center;">m</th>
                <th scope="col" class="num">λ [nm]</th>
                <th scope="col" class="num">V</th>
                <th scope="col" class="num">n<sub>e</sub></th>
                <!-- <th scope="col" class="num">A</th> -->
                <th scope="col" class="num">iB/N</th>
                <!-- <th scope="col" class="num">i ν</th> -->
            </tr>
            </thead>
            <tbody>
            <tr>
                <td style="text-align:center;">{self.mode_label()}</td>
                <td style="text-align:center;">{self.ell}</td>
                <td style="text-align:center;">{self.m}</td>
                <td class="num">{swatch_html}{wl_str}</td>
                <td class="num">{self.V:.2f}</td>
                <td class="num">{self.neff:.4f}</td>
                <!-- <td class="num">{format_complex_auto(self.A)}</td> -->
                <td class="num">{format_complex_auto(1j*self.B)}</td>
                <!-- <td class="num">{nu_html}</td> -->
            </tr>
            </tbody>
        </table>
        </div>
        """
        return html
    
    