    # def _cartesian_components(self, field, rho_flat, phi_flat, z_flat):
    #     """
    #     Returns (Fx, Fy, Fz) flat arrays for requested field ('E' or 'H'),
    #     using analytic TE/TM for ℓ=0, hybrid formalism otherwise.
    #     """
    #     A, B = self.A, self.B
    #     ell = self.ell

    #     if ell == 0:
    #         kz = self.kz
    #         ne = self.neff
       
    #         mu = self.mu(rho_flat)
    #         eps = self.eps(rho_flat)

    #         kapr = self.kap(rho_flat)/self.k0
    #         R1 = self._radial_function(+1, rho_flat)
    #         R0 = self._radial_function(0, rho_flat)
    #         # if np.allclose(z_flat, 0):
    #         #     phase = 1
    #         # else:
    #         #     phase = np.exp(1j * kz * z_flat)
    #         phase = np.exp(1j * kz * z_flat)

    #         if field.lower() == 'e':
    #             if A == 0 and B != 0:
    #                 # TE: only phi component
    #                 E_rho = np.zeros_like(rho_flat, dtype=complex)
    #                 E_phi = mu * np.abs(B) / (kapr * np.sqrt(eps0)) * R1 * phase
    #                 E_z   = np.zeros_like(rho_flat, dtype=complex)
    #                 E_cyl = np.stack([E_rho, E_phi, E_z], axis=-1)
    #                 return cylindrical_to_cartesian(E_cyl, phi_flat)
    #             elif B == 0 and A != 0:
    #                 # TM: rho, z components
    #                 E_rho = ne * np.abs(A) / (kapr) * R1 * phase
    #                 E_phi = np.zeros_like(rho_flat, dtype=complex)
    #                 E_z   = - 1j* np.abs(A) * R0 * phase
    #                 E_cyl = np.stack([E_rho, E_phi, E_z], axis=-1)
    #                 # remove numerical noise
    #                 E_cyl = (E_cyl)/ np.sqrt(eps0)
    #                 return cylindrical_to_cartesian(E_cyl, phi_flat)
    #         elif field.lower() == 'h':
    #             if A == 0 and B != 0:
    #                 # TE: rho, z components
    #                 H_rho = - ne * np.abs(B) / (kapr) * R1 * phase
    #                 H_phi = np.zeros_like(rho_flat, dtype=complex)
    #                 H_z   = 1j* np.abs(B) * R0 * phase
    #                 H_cyl = np.stack([H_rho, H_phi, H_z], axis=-1)
    #                 # remove numerical noise
    #                 H_cyl = (H_cyl)/ np.sqrt(mu0)
    #                 return cylindrical_to_cartesian(H_cyl, phi_flat)
    #             elif B == 0 and A != 0:
    #                 # TM: only phi component
    #                 H_rho = np.zeros_like(rho_flat, dtype=complex)
    #                 H_phi = eps * np.abs(A) / (kapr * np.sqrt(mu0)) * R1 * phase
    #                 H_z   = np.zeros_like(rho_flat, dtype=complex)
    #                 H_cyl = np.stack([H_rho, H_phi, H_z], axis=-1)
    #                 return cylindrical_to_cartesian(H_cyl, phi_flat)
    #     # Fallback for hybrid
    #     F0, Fp, Fm = self._spin_components(field, rho_flat, phi_flat, z_flat)
    #     return spin_to_cartesian(F0, Fp, Fm)

        # F_cartesian = self._cartesian_components('H', rho_flat, phi_flat, z_flat)
        # return F_cartesian.reshape(*shape, 3)


    
    # @staticmethod
    # def _zero_physical_noise(F, factor=4):
    #     eps = np.finfo(float).eps
    #     threshold = factor * eps
    #     F_real = np.where(np.abs(F.real) < threshold, 0.0, F.real)
    #     F_imag = np.where(np.abs(F.imag) < threshold, 0.0, F.imag)
    #     return F_real + 1j * F_imag
        



    # phi_epsJ = eps1 * (u * b * Jp) + eps2 * (w * (1-b) * DlnK * J)
    # phi_muJ = mu1 * (u * b * Jp) + mu2 * (w * (1-b) * DlnK * J)

    # if ell == 0:
    #     if mode_type is not None:
    #         mt = mode_type.lower()
    #         if mt == "te":
    #             return phi_epsJ
    #         elif mt == "tm":
    #             return phi_muJ
    #         else:
    #             raise ValueError("mode_type must be 'TE', 'TM', or None for ell=0.")
    #     else:
    #         return (phi_epsJ * phi_muJ - J ** 2 * (ell * ne) ** 2)
    # else:
    #     # return (phi_epsJ * phi_muJ - J ** 2 * (ell * ne) ** 2)
    #     # return (phi_epsJ * phi_muJ / (ell) - ell * J ** 2 * (ne) ** 2)
    #     phi_epsJ_per_V = eps1 * (sqb1 * b * Jp) + eps2 * (sqb * (1-b) * DlnK * J)
    #     phi_muJ_per_V  =  mu1 * (sqb1 * b * Jp) +  mu2 * (sqb * (1-b) * DlnK * J)
    #     # return (phi_epsJ * phi_muJ - J ** 2 * (ell * ne) ** 2)
    #     return (phi_epsJ_per_V * phi_muJ_per_V / (ell) - ell * J ** 2 * (ne/Vnum) ** 2)

    # with np.errstate(divide='ignore', invalid='ignore'):
    #     Jl_u = jv(ell, u)
    #     Jl_u_p = jvp(ell, u)
    #     Kv_w = kv(ell, w)
    #     Kv_w_p = kvp(ell, w)

    #     phi_eps = ((u * w / Vnum) ** 2) * (eps1 / u * Jl_u_p / Jl_u + eps2 / w * Kv_w_p / Kv_w)
    #     phi_mu = ((u * w / Vnum) ** 2) * (mu1 / u * Jl_u_p / Jl_u + mu2 / w * Kv_w_p / Kv_w)

    #     if ell == 0 and mode_type is not None:
    #         mt = mode_type.lower()
    #         if mt == "te":
    #             return phi_mu * Jl_u
    #         elif mt == "tm":
    #             return phi_eps * Jl_u
    #         else:
    #             raise ValueError("mode_type must be 'TE', 'TM', or None for ell=0.")
    #     else:
    #         return (phi_eps * phi_mu - (ell * ne) ** 2) * (Jl_u ** 2)
    #         # return (phi_eps * phi_mu / (ell**2) - (ne) ** 2) * (Jl_u ** 2)
    


# def F_dispersion(fibre, ell, b, V=None, wavelength=None, mode_type=None):
#     b = np.asarray(b)

#     if V is not None:
#         wl = fibre.wavelength_from_V(V)
#         Vnum = np.asarray(V)
#     elif wavelength is not None:
#         wl = wavelength
#         Vnum = fibre.V(wavelength)
#     else:
#         raise ValueError("Specify either V or wavelength.")
    
#     #replace Vnum=0 with smallest number
#     Vnum[Vnum == 0] = np.finfo(float).eps

#     n1 = fibre.n_core(wl)
#     n2 = fibre.n_clad(wl)
#     eps1 = fibre._eval(fibre.eps_core, wl)
#     eps2 = fibre._eval(fibre.eps_clad, wl)
#     mu1 = fibre._eval(fibre.mu_core, wl)
#     mu2 = fibre._eval(fibre.mu_clad, wl)

#     ne = np.sqrt(b * n1**2 + (1 - b) * n2**2)
#     sqb = np.sqrt(b)
#     sqb1 = np.sqrt(1 - b)
#     u = Vnum * sqb1
#     w = Vnum * sqb

#     J = jv(ell, u)
#     Jp = jvp(ell, u)
#     # DlnK = -(kve(ell - 1, w) + kve(ell+1, w))/(2*kve(ell, w))
#     def DlnK_safe_where(ell, w):
#         w = np.asarray(w)
#         num = -(kve(ell-1, w) + kve(ell+1, w))
#         den = 2.0 * kve(ell, w)
#         out = np.full_like(num, np.nan)
#         np.divide(num, den, out=out,
#                 where=np.isfinite(num) & np.isfinite(den) & (den != 0))
#         return out
#     DlnK = DlnK_safe_where(ell, w)

#     phi_epsJ_per_V = eps1 * (sqb1 * b * Jp) + eps2 * (sqb * (1-b) * DlnK * J)
#     phi_muJ_per_V  =  mu1 * (sqb1 * b * Jp) +  mu2 * (sqb * (1-b) * DlnK * J)

#     if ell == 0:
#         if mode_type is not None:
#             mt = mode_type.lower()
#             if mt == "te":
#                 return phi_epsJ_per_V*b*(1-b)
#             elif mt == "tm":
#                 return phi_muJ_per_V*b*(1-b)
#             else:
#                 raise ValueError("mode_type must be 'TE', 'TM', or None for ell=0.")
#         else:
#             return (phi_epsJ_per_V * phi_muJ_per_V  -  J ** 2 * (ell * ne/Vnum) ** 2)*b*(1-b)
#     else:
#         return (phi_epsJ_per_V * phi_muJ_per_V / (ell * ell) - J ** 2 * (ne/Vnum) ** 2)*b*(1-b)