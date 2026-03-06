# How It Works

`anafibre` solves guided modes of cylindrical step-index fibres by combining:

- analytical field expressions in cylindrical geometry,
- a characteristic (dispersion) equation for the effective index, and
- numerical root-finding for each mode family.

## 1. Physical Model

The fibre is modelled as a cylindrical structure with piecewise-constant material parameters (relative permittivity and permeability) that abruptly change at the boundary of core/cladding which happens at radius $\rho_0$:

\[
\alpha_\text{r}=\left\{\begin{array}{ll}
\alpha_1, & \text { for } \rho<\rho_0 \\
\alpha_2, & \text { for } \rho>\rho_0
\end{array}\qq{where}\alpha_\text{r}\in\{\varepsilon_\text{r},\mu_\text{r}\}\right.\,.
\]

The discontinuity in material parameters leads to dicontinuity of the refractive index:

\[
n=\sqrt{\varepsilon_\text{r}\mu_\text{r}}=\left\{\begin{array}{ll}
n_1, & \text {for } \rho<\rho_0 \\
n_2, & \text {for } \rho>\rho_0
\end{array}\right.\,,
\]

and if $n_1>n_2$ then the fibre can support guided modes that are confined to the core and decay evanescently in the cladding.

## 2. Modal Ansatz

We seek time-harmonic guided modes with angular frequency $\omega$ and propagation constant $k_z$ along the fibre axis, which is assumed to be the $z$-axis. Since we assume the fibre to be infinitely long, it has cylindrical symmetry and we can separate variables in cylindrical coordinates $(\rho,\varphi,z)$. Each mode should be an eigenfunction of the total angular momentum operator

\[
\hat{J}_z\vc{F}_{\ell m}={(\hat{L}_z+\hat{S}_z)}\vc{F}_{\ell m}=\hbar \ell\vc{F}_{\ell m}\,,
\]

where the azimuthal number is $\ell\in\mathbb{Z}$. This leads to the following ansatz for the electric and magnetic fields:

\[
\vc{F}_{\ell m}(t,\vc{r}) = e^{\ii(k_z z+ \ell \varphi - \omega t)}\,
\sum_{s=-1}^{1} \tilde{F}_{\ell m}^{(s)}\,\mathrm{Z}_{\ell-s}(\kappa\rho)\,
 e^{-\ii s\varphi}\,\uv{e}_s\,,
\]

where $\uv{e}_0=\uv{z}$ and $\uv{e}_{\pm1}=(\uv{x}\pm\ii\uv{y})/\sqrt{2}$ are the standard circular polarisation unit vectors, and $e^{\mp\ii \varphi}\,\uv{e}_{\pm1}=(\uv{\rho}\pm\ii\uv{\varphi})/\sqrt{2}$ are the corresponding cylindrical spin-weighted unit vectors. The radial dependence, $\mathrm{Z}_{\ell-s}(\kappa\rho)$, satisfies the Bessel equation:

\[
    (\kappa\rho)^2\mathrm{Z}''_{\ell-s}+\kappa\rho\mathrm{Z}'_{\ell-s}+[(\kappa\rho)^2-(\ell-s)^2]\mathrm{Z}_{\ell-s}=0\,,
\]

where $\kappa^2=k^2-k_z^2$ is the transverse wavenumber and primes denote differentiation with respect to the argument. For guided modes, we choose normalised solutions:

\[
        \mathrm{Z}_{\ell-s}(\kappa\rho)=
    \begin{cases}
        {\bessel{\ell-s}(\kappa_1\rho)}/{\bessel{\ell}(\kappa_1\rho_0)}& \text{for} \quad\rho<\rho_0
        \\{\hankel{\ell-s}(\kappa_2\rho)}/{\hankel{\ell}(\kappa_2\rho_0)}& \text{for} \quad\rho>\rho_0
    \end{cases}\;,
    \]

where $\bessel{\ell}$ and $\hankel{\ell}$ are Bessel and Hankel functions of the first kind, respectively. The normalisation ensures that the longitudinal components of the fields are automatically continuous at the core/cladding boundary $\rho=\rho_0$. The complex coefficients $\tilde{F}_{\ell m}^{(s)}$ can be all related to longitudinal amplitudes $A_{\ell m}$ and $B_{\ell m}$, and the transverse components are then determined by Maxwell's equations

\[
        \curl\vc{F}_\text{e}=\ii k_0\mu_\text{r}\vc{F}_\text{m}\,,\quad
    \curl\vc{F}_\text{m}=-\ii k_0\varepsilon_\text{r}\vc{F}_\text{e}\,,
    \]

using the spin-raising/lowering operators $\partial_{\pm1}=(\uv{e}_{\pm1}^*\vdot\grad)=(\partial_{x}\mp\ii\partial_{y})/\sqrt{2}$,

\[
\uv{e}_{\pm1}^*\vdot(\curl\vc{F})=\pm\ii\big[\partial_{\pm1}(\uv{e}_0\vdot\vc{F})-\partial_{z}(\uv{e}_{\pm1}^*\vdot\vc{F})\big]\,.
\]

These operators act on the spin-weighted cylindrical functions as

\[
\partial_{\pm1}[\harmol{\ell}(\kappa\rho)\ee^{\ii \ell\varphi}]=\pm\frac{\kappa}{\sqrt{2}}\harmol{\ell\mp1}(\kappa\rho)\ee^{\ii (\ell\mp1)\varphi}\,,
\]

which leads to the following relations between the coefficients:

\[
    \begin{alignedat}{2}
    \tilde{E}_{\ell m}^{(0)}&=\frac{A_{\ell m}}{\sqrt{\varepsilon_0}}\,,&\quad\tilde{E}_{\ell m}^{(\pm1)}&=\frac{\mathop{\pm} \ii n_\text{e} A_{\ell m}-\mu_\text{r} B_{\ell m}}{\sqrt{2\varepsilon_0(n^2-n_\text{e}^2)}}\,,\\
    \tilde{H}_{\ell m}^{(0)}&=\frac{B_{\ell m}}{\sqrt{\mu_0}}\,,&\quad\tilde{H}_{\ell m}^{(\pm1)}&=\frac{\mathop{\pm} \ii n_\text{e} B_{\ell m}+\varepsilon_\text{r} A_{\ell m}}{\sqrt{2\mu_0(n^2-n_\text{e}^2)}}\,.,
    \end{alignedat}
\]

where $n_\text{e}=k_z/k_0$ is the effective index of the mode. The dispersion equation for $n_\text{e}$ is obtained by enforcing the continuity of the azimuthal fields at $\rho=\rho_0$.

## 3. Dispersion Equation

To turn the field ansatz into an eigenvalue problem, we enforce continuity of the tangential field components at $\rho=\rho_0$. This yields a scalar characteristic equation in terms of the normalised propagation parameter

\[
b=\frac{n_\mathrm{e}^2-n_2^2}{n_1^2-n_2^2},\qquad 0<b<1\quad\text{(guided, lossless case)}.
\]

The standard transverse parameters are

\[
u=V\sqrt{1-b},\qquad w=V\sqrt{b},\qquad V=k_0\rho_0\sqrt{n_1^2-n_2^2},
\]

and in `anafibre` the hybrid auxiliary functions are

\[
\Phi_\ell^\alpha(b,V)=\left(\frac{uw}{V}\right)^2
\left[\frac{\alpha_1}{u}\frac{\bessel{\ell}^{\prime}(u)}{\bessel{\ell}(u)}
+\frac{\alpha_2}{w}\frac{\macdonald{\ell}^{\prime}(w)}{\macdonald{\ell}(w)}\right],
\]

with $\alpha\in\{\varepsilon,\mu\}$ (so $\alpha_1,\alpha_2$ are core/cladding values), $\bessel{\ell}$ the Bessel function of the first kind, and $\macdonald{\ell}$ the modified Bessel (Macdonald) function.

For $\ell\neq 0$ (hybrid modes), the dispersion condition is

\[
\Phi_\ell^\varepsilon\,\Phi_\ell^\mu-(\ell n_\mathrm{e})^2=0,
\qquad
n_\mathrm{e}=\sqrt{b\,n_1^2+(1-b)\,n_2^2}.
\]

For $\ell=0$, the branches decouple into

\[
\text{TE}_{0m}:\ \Phi_0^\mu=0,
\qquad
\text{TM}_{0m}:\ \Phi_0^\varepsilon=0.
\]

In implementation, root-finding uses a regularized residual `F(\ell,b,V)` (see `anafibre.dispersion.F_dispersion` / `StepIndexFibre.F`) that has the same zeros as the physical equation but better numerical behaviour:

\[
F=\begin{cases}
\Phi_0^\mu \bessel{0}(u)\,\sqrt{b(1-b)} & \text{for TE},\\
\Phi_0^\varepsilon \bessel{0}(u)\,\sqrt{b(1-b)} & \text{for TM},\\
\big[\Phi_\ell^\varepsilon\Phi_\ell^\mu-(\ell n_\mathrm{e})^2\big]\bessel{\ell}^2(u)\,\dfrac{b(1-b)}{\ell^2} & \text{for }\ell\neq 0.
\end{cases}
\]

The extra factors remove removable singular behaviour near cutoffs and Bessel zeros, making bracketed root searches more stable without changing the physical mode spectrum.

## 4. From Roots To Modes

For a chosen wavelength:

1. Compute fibre/material quantities and V-number.
2. Solve the characteristic equation for admissible $n_\mathrm{e}$ roots.
3. Label roots by family/order ($\text{TE}_{0m}$, $\text{TM}_{0m}$, $\text{HE}_{\ell m}$, $\text{EH}_{\ell m}$).
4. Construct a `GuidedMode` object with chosen polarisation coefficients.

This is exposed through `StepIndexFibre` methods such as `HE`, `EH`, `TE`, `TM`, and `list_modes_at`.

## 5. Field Reconstruction

Once a mode is constructed, `GuidedMode` reconstructs vector fields analytically and evaluates them at points or arrays:

- `E(x,y,z)`, `H(x,y,z)`
- `gradE(...)`, `gradH(...)`
- `Power(...)`
- Stokes quantities `S0`, `S1`, `S2`, `S3`

So dispersion-level quantities (`n_eff`, `k_z`, ...) and full spatial fields are produced in one consistent model.

## 6. Practical Notes

- Inputs are SI by default (meters, seconds, etc.).
- `n`, `eps`, and `mu` may be constants or wavelength-dependent callables.
- Hybrid modes depend on user-selected amplitude/polarisation coefficients.
- Plotting/animation helpers in `anafibre.plotting` operate directly on `GuidedMode` outputs.
