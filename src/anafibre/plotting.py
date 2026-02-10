"""
plotting.py

Functions for visualizing dispersion relations, mode profiles, and wavelength spectra.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn_zeros
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib.colors import Normalize, ListedColormap, hsv_to_rgb
from .utils import wavelength_to_rgb
from .dispersion import F_dispersion
from .utils import units, _HAS_UNITS


def plot_dispersion_chart(
    fibre,
    ell=1,
    Vmin=0, Vmax=10, Npoints=500,
    bmin=0, bmax=1,
    mode_type=None,
    show_bessel_zeros=True,
    colorbar=False,
    ax=None,
    cbar_label=r'$F_\ell(b,V)$',
    show_xgrid=False,
    show_bessel_grid=True,
    bessel_grid_kwargs=None,
    ylabel=True,
    xlabel=True, rasterized=True
):
    V_vals = np.linspace(Vmin, Vmax, Npoints)
    b_vals = np.linspace(bmin, bmax, Npoints)
    VV, BB = np.meshgrid(V_vals, b_vals)

    F_grid = F_dispersion(fibre, ell=ell, b=BB, V=VV, mode_type=mode_type)

    absmax = np.nanmax(np.abs(F_grid[np.isfinite(F_grid)]))
    if np.isnan(absmax):
        # Choose defaults or skip the plot
        vmin, vmax = None, None
    else:
        vmax = absmax
        vmin = -vmax
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4.5))
    else:
        fig = ax.figure

    im = ax.pcolormesh(V_vals, b_vals, F_grid, cmap='RdBu_r', vmin=vmin, vmax=vmax, shading="auto")
    cs = ax.contour(V_vals, b_vals, F_grid, levels=[0], colors='k', linewidths=1, linestyles='-')
    im.set_rasterized(rasterized)

    if xlabel:
        ax.set_xlabel(r"$V=\rho_0\sqrt{k_1^2 - k_2^2}$")
    if ylabel:
        ax.set_ylabel(r"$b = (k_z^2-k_2^2)/(k_1^2-k_2^2)$")
    ax.set_xlim(Vmin, Vmax)
    ax.set_ylim(bmin, bmax)
    ax.set_title(rf"$\ell = {ell}$")

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.25)
        cb = fig.colorbar(im, cax=cax, orientation='vertical', label=cbar_label)
        cb.set_ticks([vmin, 0, vmax])
        cb.set_ticklabels(['min', 0, 'max'])
        cb.ax.tick_params(labelsize=10)

    zeros_in_range = np.array([])
    secax = None
    if show_bessel_zeros:
        num_zeros = int((Vmax - Vmin) // np.pi) + 5
        zeros = jn_zeros(ell, num_zeros)
        zeros_in_range = zeros[(zeros >= Vmin) & (zeros <= Vmax)]
        secax = ax.secondary_xaxis('top', functions=(lambda V: V, lambda V: V))
        secax.set_xlim(Vmin, Vmax)
        secax.set_xticks(zeros_in_range)
        secax.set_xticklabels([rf"$j_{{{ell},{i+1}}}$" for i in range(len(zeros_in_range))])

    if show_xgrid:
        ax.grid(axis='x', linestyle=':', color='gray', alpha=0.5)

    if show_bessel_grid and len(zeros_in_range) > 0:
        line_style = dict(color="gray", linestyle="--", linewidth=.5, alpha=0.6, zorder=2)
        if bessel_grid_kwargs:
            line_style.update(bessel_grid_kwargs)
        for x in zeros_in_range:
            ax.axvline(x, **line_style)

    plt.tight_layout()
    return ax

def plot_dispersion_vs_wavelength(
    fibre,
    ell=1,
    wavelength_min=400.0, wavelength_max=700.0, Npoints=500,    # in nanometers
    bmin=0, bmax=1,
    mode_type=None,
    show_bessel_zeros=True,
    colorbar=False,
    ax=None,
    cbar_label=r'$F_\ell(b,\lambda)$',
    show_xgrid=False,
    show_bessel_grid=True,
    bessel_grid_kwargs=None,
    ylabel=True,
    xlabel=True,
    rasterized=True
):
    """
    Same visualization as plot_dispersion_chart but with the horizontal axis
    in wavelength (nm) instead of V. Wavelengths are sampled linearly
    between wavelength_min and wavelength_max (nm).

    The function attempts to convert wavelength -> V using either:
        • fibre.V_from_wavelength(wavelength_in_m) or fibre.V_of_wavelength(...)
        • or via core radius and refractive indices:
            V(λ) = 2π * a / λ * sqrt(n_core(λ)^2 - n_clad(λ)^2)
        Where `a` is fibre.core_radius or fibre.rho_0 and n_core/n_clad
        can be callables (n(λ) ) or scalars.
    """

    # Build wavelength array (meters)
    lam_nm = np.linspace(float(wavelength_min), float(wavelength_max), int(Npoints))
    lam_m = lam_nm * 1e-9

    # Helper: try user-provided conversion methods first
    def _V_from_wavelength_try(lam):
        # lam in meters, lam can be array
        if hasattr(fibre, "V_from_wavelength"):
            return np.asarray(fibre.V_from_wavelength(lam))
        if hasattr(fibre, "V_of_wavelength"):
            return np.asarray(fibre.V_of_wavelength(lam))
        return None

    V_try = None
    try:
        V_try = _V_from_wavelength_try(lam_m)
    except Exception:
        V_try = None

    # Fallback: analytic conversion if possible: V = 2π * a / λ * sqrt(n1^2 - n2^2)
    def _compute_V_via_indices(lam):
        # lam : array in meters
        # find core radius
        a = None
        if hasattr(fibre, "core_radius"):
            a = fibre.core_radius
        elif hasattr(fibre, "rho_0"):
            a = fibre.rho_0
        # handle astropy Quantity
        try:
            if hasattr(a, "to"):
                a_val = float(a.to(units.m).value)
            else:
                a_val = float(a)
        except Exception:
            a_val = None

        # refractive indices: prefer callables n_core(λ) and n_clad(λ) if available
        n_core_obj = getattr(fibre, "n_core", None)
        n_clad_obj = getattr(fibre, "n_clad", None)
        n1_obj = getattr(fibre, "n1", None)
        n2_obj = getattr(fibre, "n2", None)

        n_core = n_core_obj or n1_obj
        n_clad = n_clad_obj or n2_obj

        if a_val is None or n_core is None or n_clad is None:
            return None

        # Evaluate n(λ). Accept scalars or callables. If callable, pass λ in meters.
        def _eval_n(nobj, lam):
            if callable(nobj):
                try:
                    return np.asarray(nobj(lam))
                except Exception:
                    # try with nanometers
                    return np.asarray(nobj(lam * 1e9))
            else:
                return np.full_like(lam, float(nobj), dtype=float)

        n_core_vals = _eval_n(n_core, lam)
        n_clad_vals = _eval_n(n_clad, lam)

        # ensure arrays
        lam = np.asarray(lam, dtype=float)
        V = 2.0 * np.pi * a_val / lam * np.sqrt(np.maximum(0.0, n_core_vals**2 - n_clad_vals**2))
        return V

    if V_try is None:
        V_vals = _compute_V_via_indices(lam_m)
    else:
        V_vals = np.asarray(V_try)

    if V_vals is None:
        raise ValueError(
            "Unable to convert wavelength -> V. Provide fibre.V_from_wavelength or attributes "
            "core_radius and n_core/n_clad (or n1/n2), where n_* may be scalars or callables."
        )

    # Build b grid and mesh for F_dispersion
    b_vals = np.linspace(bmin, bmax, int(Npoints))
    WW, BB = np.meshgrid(lam_m, b_vals)      # WW: meters (x axis), BB: b
    # Need corresponding V mesh
    # broadcast V_vals (shape (Npoints,)) across rows to match WW shape
    VV = np.tile(V_vals.reshape(1, -1), (b_vals.size, 1))

    # evaluate F_dispersion (expects V in same units as used elsewhere)
    F_grid = F_dispersion(fibre, ell=ell, b=BB, V=VV, mode_type=mode_type)

    absmax = np.nanmax(np.abs(F_grid[np.isfinite(F_grid)]))
    if np.isnan(absmax):
        vmin, vmax = None, None
    else:
        vmax = absmax
        vmin = -vmax

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4.5))
    else:
        fig = ax.figure

    # pcolormesh with wavelength axis in nanometers for nicer ticks
    ax.pcolormesh(lam_nm, b_vals, F_grid, cmap='RdBu_r', vmin=vmin, vmax=vmax, shading="auto", rasterized=rasterized)
    ax.contour(lam_nm, b_vals, F_grid, levels=[0], colors='k', linewidths=1, linestyles='-')

    if xlabel:
        ax.set_xlabel("Wavelength (nm)")
    if ylabel:
        ax.set_ylabel(r"$b = (k_z^2-k_2^2)/(k_1^2-k_2^2)$")
    ax.set_xlim(lam_nm[0], lam_nm[-1])
    ax.set_ylim(bmin, bmax)
    ax.set_title(rf"$\ell = {ell}$")

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.25)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap='RdBu_r'), cax=cax, orientation='vertical', label=cbar_label)
        try:
            cb.set_ticks([vmin, 0, vmax])
            cb.set_ticklabels(['min', 0, 'max'])
        except Exception:
            pass
        cb.ax.tick_params(labelsize=10)

    zeros_in_range = np.array([])
    if show_bessel_zeros:
        # compute bessel zeros in V-space (as in the original) and attempt to map to wavelengths.
        num_zeros = int((np.max(V_vals) - np.min(V_vals)) // np.pi) + 5
        try:
            zeros = jn_zeros(ell, num_zeros)
            zeros_in_V = zeros[(zeros >= V_vals.min()) & (zeros <= V_vals.max())]
            # map each zero V0 to an approximate wavelength.
            lam_from_zero = []
            # Try direct inversion assuming locally-constant refractive indices:
            # λ ≈ 2π * a * sqrt(n_core^2 - n_clad^2) / V0
            lam_approx = None
            try:
                # use mid-wavelength reference to estimate mapping
                mid = np.mean(lam_m)
                V_ref = _compute_V_via_indices(np.array([mid]))
                if V_ref is not None and V_ref.size == 1 and V_ref[0] > 0:
                    # compute factor f = λ * V / (2π a sqrt(...)) -> we invert directly below
                    # implement approximate inversion: λ ≈ 2π * a * sqrt(n_core(mid)^2 - n_clad(mid)^2) / V0
                    lam_approx_vals = []
                    a = None
                    if hasattr(fibre, "core_radius"):
                        a = fibre.core_radius
                    elif hasattr(fibre, "rho_0"):
                        a = fibre.rho_0
                    try:
                        a_val = float(a.to(units.m).value) if hasattr(a, "to") else float(a)
                    except Exception:
                        a_val = None
                    if a_val is not None:
                        n_core_ref = None
                        n_clad_ref = None
                        n_core_obj = getattr(fibre, "n_core", None) or getattr(fibre, "n1", None)
                        n_clad_obj = getattr(fibre, "n_clad", None) or getattr(fibre, "n2", None)
                        if n_core_obj is not None and n_clad_obj is not None:
                            try:
                                n_core_ref = (n_core_obj(mid) if callable(n_core_obj) else float(n_core_obj))
                                n_clad_ref = (n_clad_obj(mid) if callable(n_clad_obj) else float(n_clad_obj))
                                for V0 in zeros_in_V:
                                    lam_approx_vals.append(2.0*np.pi*a_val*np.sqrt(max(0.0, n_core_ref**2 - n_clad_ref**2)) / V0)
                                lam_from_zero = np.array(lam_approx_vals) * 1e9  # to nm
                            except Exception:
                                lam_from_zero = np.array([])
            except Exception:
                lam_from_zero = np.array([])

            # Keep only those that fall in wavelength range
            if lam_from_zero.size > 0:
                lam_lines_nm = lam_from_zero[(lam_from_zero >= lam_nm[0]) & (lam_from_zero <= lam_nm[-1])]
            else:
                lam_lines_nm = np.array([])

            # Add top axis ticks for labelled zeros (use approximate mapping)
            if lam_lines_nm.size > 0:
                secax = ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
                secax.set_xlim(lam_nm[0], lam_nm[-1])
                secax.set_xticks(lam_lines_nm)
                secax.set_xticklabels([rf"$j_{{{ell},{i+1}}}$" for i in range(len(lam_lines_nm))])
                # draw vertical grid lines at these approx wavelengths if requested
                if show_bessel_grid:
                    line_style = dict(color="gray", linestyle="--", linewidth=.5, alpha=0.6, zorder=2)
                    if bessel_grid_kwargs:
                        line_style.update(bessel_grid_kwargs)
                    for x in lam_lines_nm:
                        ax.axvline(x, **line_style)
        except Exception:
            # ignore bessel-zero plotting errors
            pass

    if show_xgrid:
        ax.grid(axis='x', linestyle=':', color='gray', alpha=0.5)

    plt.tight_layout()
    return ax

def add_visible_spectrum(ax, position=[0, 0, 1, 0.01], wavelengths=np.linspace(400, 700, 300), xlim=None, resolution=300):
    """
    Adds a visible spectrum color strip below the x-axis of a given plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis where the strip will be added.
    position : list of 4 floats
        Inset position [x, y, width, height] in axes coordinates.
    wavelengths : array-like
        Wavelengths in nanometers to map to RGB.
    xlim : tuple, optional
        If provided, overrides the wavelength range.
    resolution : int
        Number of color points.
    """
    if xlim is not None:
        wavelengths = np.linspace(xlim[0], xlim[1], resolution)

    colors = [wavelength_to_rgb(w) for w in wavelengths]

    ax2 = ax.inset_axes(position, transform=ax.transAxes)
    ax2.imshow([colors], aspect='auto', extent=[wavelengths[0], wavelengths[-1], 0, 1])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.axis('off')


def plot_nu_vs_V(
    fibre, ell, m, *, Vmin=0.0, Vmax=10.0, Npoints=400, mode_type=None,
    ax=None, show=('real', 'imag'), title=None
):
    """
    Plot ν(V) for a given (ell, m), following the definition used in fields.py.

    show: tuple containing any of {'real','imag','abs','angle'} to display.
    Returns the axes.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    V = np.linspace(Vmin, Vmax, int(Npoints))
    nu, mask = fibre.nu_vs_V(ell, m, V, mode_type=mode_type)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    Vv = V[mask]
    nuv = nu[mask]

    show = tuple(s.lower() for s in show)
    if 'real' in show:
        ax.plot(Vv, np.real(nuv), label=r'Re($\nu$)')
    if 'imag' in show:
        ax.plot(Vv, np.imag(nuv), '--', label=r'Im($\nu$)')
    if 'abs' in show:
        ax.plot(Vv, np.abs(nuv), ':', label=r'|$\nu$|')
    if 'angle' in show:
        ax.plot(Vv, np.angle(nuv), '-.', label=r'$\angle \nu$')

    ax.set_xlim(Vmin, Vmax)
    ax.set_xlabel('V')
    ax.set_ylabel(r'$\nu$')
    # if title is None:
    #     ax.set_title(f"$\nu(V) for \ell={ell}, m={m}{' ('+str(mode_type)+')' if mode_type else ''}$")
    # else:
    #     ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    return ax


def plot_sigma_vs_V(
    fibre, ell, m, *, Vmin=0.0, Vmax=10.0, Npoints=400, mode_type=None,
    ax=None, title=None
):
    """
    Plot σ(V) used for modal power normalisation for a given (ell, m).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    V = np.linspace(Vmin, Vmax, int(Npoints))
    sigma, mask = fibre.sigma_vs_V(ell, m, V, mode_type=mode_type)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    ax.plot(V[mask], sigma[mask], label=r"$\sigma(V)$")
    ax.set_xlim(Vmin, Vmax)
    ax.set_xlabel('V')
    ax.set_ylabel(r"$\sigma$ (SI)")
    # if title is None:
    #     ax.set_title(f"σ(V) for ℓ={ell}, m={m}{' ('+str(mode_type)+')' if mode_type else ''}")
    # else:
    #     ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    return ax


def plot_xy_vector_field(
    X, Y, F, ax=None, scale="auto", zscale=None, cmap='RdBu_r', stride=4, title=None,
    xlabel=None, ylabel=None, colorbar_label=None, colorbar=False,
    density=1.0, linewidth=1, color='k', type='quiver', labels=False
):
    """
    Plot XY vector field and colormap, showing both inside and outside fibre.
    X, Y : 2D arrays (SI units or astropy Quantity)
    F    : (..., 3) vector field (real or Quantity)

    Parameters
    ----------
    scale : {"auto", float}
        Relative multiplier applied to the automatically determined quiver scale
        from the transverse field amplitude. Use "auto" or None for factor 1.0;
        provide a number (e.g., 0.7 or 1.5) to shrink/enlarge arrows w.r.t. autoscale.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Get axis units for labels
    def _get_unit_and_val(arr):
        if hasattr(arr, 'unit'):
            return arr.unit, arr.value
        return None, arr

    x_unit, Xv = _get_unit_and_val(X)
    y_unit, Yv = _get_unit_and_val(Y)

    # # zero out small values
    # F = np.where(np.abs(F) < 1e-15, 0.0, F)


    Fx, Fy, Fz = F[..., 0], F[..., 1], F[..., 2]
    Fxv = Fx.real.value if hasattr(Fx, 'unit') else Fx.real
    Fyv = Fy.real.value if hasattr(Fy, 'unit') else Fy.real
    Fzv = Fz.real.value if hasattr(Fz, 'unit') else Fz.real

    Fxv[Fxv == 0] = 0.0
    Fyv[Fyv == 0] = 0.0

    # Compute transverse field amplitude for auto-scaling
    transverse_mag = np.sqrt(Fxv**2 + Fyv**2)
    finite_transverse = transverse_mag[np.isfinite(transverse_mag)]
    # Use robust percentile for scaling to avoid outliers
    auto_scale = 40
    if finite_transverse.size > 0:
        robust_max = np.percentile(finite_transverse, 99)
        if robust_max > 0:
            auto_scale = 1.0 / robust_max * 20  # 20 is a visual factor, adjust as needed

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Colormap for Fz (real part, no masking)
    if zscale is not None:
        absmax = zscale
    else:
        absmax = np.max(np.abs(Fzv))
    if absmax != 0:
        c = ax.pcolormesh(Xv, Yv, Fzv, shading='auto', cmap=cmap,
                      vmin=-absmax, vmax=absmax)
        if colorbar:
            cb_label = colorbar_label or r"$F_z$"
            plt.colorbar(c, ax=ax, label=cb_label)

    if type == 'quiver':
        # Quiver (arrows), subsampled
        # Treat `scale` as a relative multiplier with respect to `auto_scale`.
        if (scale == "auto") or (scale is None):
            scale_factor = 1.0
        else:
            try:
                scale_factor = float(scale)
            except Exception:
                scale_factor = 1.0
        quiver_scale = auto_scale * scale_factor
        ax.quiver(
            Xv[::stride, ::stride], Yv[::stride, ::stride],
            Fxv[::stride, ::stride], Fyv[::stride, ::stride],
            scale=quiver_scale, color='black', pivot='middle', linewidth=0.5, 
        )
    elif type == 'streamplot':
        # Streamlines
        ax.streamplot(Xv, Yv, Fxv, Fyv, density=density, color=color, linewidth=linewidth, arrowsize=1, broken_streamlines=False)

    if labels:
        # Add labels for the axes
        x_unit_str = f"[{x_unit:latex_inline}]" if x_unit else "[m]"
        y_unit_str = f"[{y_unit:latex_inline}]" if y_unit else "[m]"
        ax.set_xlabel(xlabel or f"x {x_unit_str}")
        ax.set_ylabel(ylabel or f"y {y_unit_str}")
    # # Labels
    # unit_str = f"[{x_unit:latex_inline}]" if x_unit else "[m]"
    # ax.set_xlabel(xlabel or f"x {unit_str}")
    # ax.set_ylabel(ylabel or f"y {unit_str}")
    ax.set_xlim(np.min(Xv), np.max(Xv))
    ax.set_ylim(np.min(Yv), np.max(Yv))
    if title:
        ax.set_title(title)
    ax.set_aspect('equal')
    return ax

# ---------- utilities ----------
def _rolled_cmap(cmap, shift=-0.5):
    """Roll a cyclic colormap by 'shift' in [0,1). For HSV, shift=-0.5 puts red at the 0 tick."""
    x = np.linspace(0, 1, 256, endpoint=False)
    return ListedColormap(cmap((x + shift) % 1.0))

def _values_and_unit(arr, target_unit=None):
    """
    Return (values, unit_or_None) for an array or astropy Quantity.
    If target_unit is given and compatible, convert to it.
    """
    try:
        q = arr.to(target_unit) if target_unit is not None else arr
        return np.asarray(q.value), q.unit
    except Exception:
        return np.asarray(arr), None

def _axis_label_from_unit(name, unit):
    # """Build a LaTeX axis label like $x\,[\mu\mathrm{m}]$ from an astropy Unit (or $x$ if dimensionless/None)."""
    if unit is None or unit == units.dimensionless_unscaled or unit == units.one:
        return rf"$\mathit{{{name}}}$"
    unit_str = unit.to_string('latex').replace('$', '')
    return rf"$\mathit{{{name}}}\,[{unit_str}]$"

# ---------- main API ----------
def plot_complex_field(
    field,
    X=None,
    Y=None,
    *,
    ax=None,
    variant="dark",                    # "dark" (s=1, v=|F|) or "light" (s=|F|^γ, v=1)
    gamma=1.0,                         # used only for variant="light"
    percentile=99.0,                   # robust scaling for |field|
    vmax=None,                         # override auto scaling if given
    show_phase_cbar=True,
    show_mag_cbar=True,
    phase_ticks=(-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi),
    phase_cmap=plt.cm.hsv,
    mag_cmap="gray",
    coord_unit=units.um,               # if set (e.g. units.um) convert X,Y to this; otherwise keep their units
    title=None,
    xlabel=None,                       # if None, auto-built from X unit
    ylabel=None,                       # if None, auto-built from Y unit
    phase_label=r"$\mathrm{Phase}\;[\mathrm{rad}]$",
    mag_label=r"$|\,\mathrm{field}\,|$",
    cbar_pad_phase=0.0,
    cbar_pad_mag=0.6,
    use_tex=False,                     # set True to use full LaTeX (requires TeX installed)
):
    """
    Plot a complex scalar field with colour = phase (0 rad → red) and brightness = magnitude.
    Returns (fig, ax, cb_phase, cb_mag).
    """
    # LaTeX / mathtext setup
    if use_tex:
        mpl.rcParams["text.usetex"] = True
        # plt.rcParams['text.usetex'] = True
        # mpl.rcParams["axes.formatter.use_mathtext"] = False
        mpl.rcParams.update({"font.family": "serif"})
    else:
        mpl.rcParams["text.usetex"] = False
        # plt.rcParams['text.usetex'] = False
        mpl.rcParams["axes.formatter.use_mathtext"] = True

    # Mask handling
    F = field

    # Magnitude/phase
    mag = np.abs(F).astype(float)
    phase = np.angle(F)  # [-pi, pi]

    # Robust magnitude scaling
    finite_mag = mag[np.isfinite(mag)]
    auto_vmax = np.percentile(finite_mag, percentile) if finite_mag.size else 1.0
    vmax_eff = (
        float(vmax) if (vmax is not None and np.isfinite(vmax) and vmax > 0)
        else (float(auto_vmax) if np.isfinite(auto_vmax) and auto_vmax > 0 else 1.0)
    )
    val = np.clip(mag / vmax_eff, 0.0, 1.0)

    # Phase → Hue (0 rad → red; cyclic)
    hue = np.mod(phase / (2*np.pi), 1.0)

    # Magnitude → HSV depending on variant
    if str(variant).lower() == "light":
        s = val**float(gamma)         # saturation encodes |field|
        v = np.ones_like(val)         # constant brightness
    else:  # "dark"
        s = np.ones_like(val)         # full saturation
        v = val                       # value encodes |field|

    rgb = hsv_to_rgb(np.stack([hue, s, v], axis=-1))

    # Axes and extent
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=False)
        created_fig = True
    else:
        fig = ax.figure

    if X is not None and Y is not None:
        xv, xunit = _values_and_unit(X, coord_unit)
        yv, yunit = _values_and_unit(Y, coord_unit)
        extent = [xv.min(), xv.max(), yv.min(), yv.max()]
        # Auto axis labels from units unless user supplied them
        if xlabel is None:
            xlabel = _axis_label_from_unit("x", xunit)
        if ylabel is None:
            ylabel = _axis_label_from_unit("y", yunit)
    else:
        ny, nx = rgb.shape[:2]
        extent = [0, nx, 0, ny]
        if xlabel is None: xlabel = r"$\mathit{x}$"
        if ylabel is None: ylabel = r"$\mathit{y}$"

    ax.imshow(rgb, origin="lower", extent=extent, interpolation="nearest")
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    if title:  ax.set_title(title)

    # Colourbars
    divider = make_axes_locatable(ax)
    cb_phase = cb_mag = None

    if show_phase_cbar:
        cax_phase = divider.append_axes("right", size="5%", pad=cbar_pad_phase)
        phase_norm = Normalize(vmin=-np.pi, vmax=np.pi)
        cmap_phase = _rolled_cmap(phase_cmap, shift=-0.5)  # centre tick (0) is red
        sm_phase = plt.cm.ScalarMappable(norm=phase_norm, cmap=cmap_phase)
        sm_phase.set_array([])
        cb_phase = fig.colorbar(sm_phase, cax=cax_phase)
        cb_phase.set_label(phase_label)
        cb_phase.set_ticks(list(phase_ticks))
        if tuple(phase_ticks) == (-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi):
            cb_phase.set_ticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])

    if show_mag_cbar:
        cax_mag = divider.append_axes("right", size="5%", pad=cbar_pad_mag)
        mag_norm = Normalize(vmin=0.0, vmax=vmax_eff)
        sm_mag = plt.cm.ScalarMappable(norm=mag_norm, cmap=mag_cmap)
        sm_mag.set_array([])
        cb_mag = fig.colorbar(sm_mag, cax=cax_mag)
        cb_mag.set_label(mag_label)
        cb_mag.formatter.set_powerlimits((0, 0))
        cb_mag.update_ticks()

    if created_fig:
        plt.show()

    return fig, ax, cb_phase, cb_mag


def plot_complex_field_polar(
    field,
    Rho,
    Phi,
    *,
    ax=None,
    variant="dark",                    # "dark" (s=1, v=|F|) or "light" (s=|F|^γ, v=1)
    gamma=1.0,                         # used only for variant="light"
    percentile=99.0,                   # robust scaling for |field|
    vmax=None,                         # override auto scaling if given
    show_phase_cbar=True,
    show_mag_cbar=True,
    phase_cbar_style="wheel",           # "bar", "wheel" (inset), or "edge" (ring around axis)
    phase_ticks=(-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi),
    phase_cmap=plt.cm.hsv,
    mag_cmap="gray",
    coord_unit=units.um,               # unit for radial coordinate
    core_radius=None,                  # if provided, set rticks at integer multiples of this
    title=None,
    rlabel=None,                       # radial axis label (auto if None)
    phase_label=r"Phase [rad]",
    mag_label=r"$|\\,\\mathrm{field}\\,|$",
    cbar_pad=0.04,
    cbar_shrink=0.7,
    cbar_fraction=0.05,               # thickness of colorbars relative to axes
    show_grid=True,                   # hide grid by default for a cleaner look
    grid_kwargs=None,                  # customize grid if shown
    phase_wheel_pos=(1, 1, 0.2, 0.2),  # (x,y,w,h) in axes fraction for wheel (style='wheel')
    phase_wheel_res=256,               # resolution of the wheel image (style='wheel')
    phase_edge_width=0.08,             # relative thickness of edge ring (style='edge')
    phase_edge_gap=0.01,               # small gap from edge (style='edge')
    phase_edge_theta_N=720,            # angular resolution (style='edge')
    phase_edge_r_N=2,                  # radial resolution (style='edge')
    use_tex=False,
):
    """
    Plot a complex scalar field defined on a polar grid using polar coordinates.

    Parameters
    ----------
    field : 2D complex array
        Complex scalar field sampled on (Rho, Phi) grid.
    Rho, Phi : array-like
        Polar grid coordinates. Either both (Nr x Nphi) arrays from meshgrid,
        or 1D arrays with lengths Nr and Nphi respectively. Rho may be an
        astropy Quantity; Phi may be a Quantity with angle units (converted to rad).

    Returns: (fig, ax, cb_phase, cb_mag)
    """
    # LaTeX / mathtext setup
    if use_tex:
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams.update({"font.family": "serif"})
        mpl.rcParams["axes.formatter.use_mathtext"] = True
    else:
        mpl.rcParams["text.usetex"] = False
        mpl.rcParams["axes.formatter.use_mathtext"] = True

    # Convert inputs and compute HSV mapping (phase→hue, magnitude→value/saturation)
    F = np.asarray(field)
    mag = np.abs(F).astype(float)
    phase = np.angle(F)

    finite_mag = mag[np.isfinite(mag)]
    auto_vmax = np.percentile(finite_mag, percentile) if finite_mag.size else 1.0
    vmax_eff = (
        float(vmax) if (vmax is not None and np.isfinite(vmax) and vmax > 0)
        else (float(auto_vmax) if np.isfinite(auto_vmax) and auto_vmax > 0 else 1.0)
    )
    val = np.clip(mag / vmax_eff, 0.0, 1.0)

    hue = np.mod(phase / (2*np.pi), 1.0)
    if str(variant).lower() == "light":
        s = val**float(gamma)
        v = np.ones_like(val)
    else:
        s = np.ones_like(val)
        v = val

    rgb = hsv_to_rgb(np.stack([hue, s, v], axis=-1))  # shape (Nr, Nphi, 3)

    # Handle NaNs by making them transparent
    alpha = np.where(np.isfinite(mag), 1.0, 0.0)
    if rgb.shape[:2] != alpha.shape:
        alpha = np.ones(rgb.shape[:2])
    rgba = np.dstack([rgb, alpha])  # (Nr, Nphi, 4)

    # Axes creation
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5, 5), constrained_layout=False)
        created_fig = True
    else:
        fig = ax.figure

    # Coordinates: convert Rho to coord_unit, Phi to radians
    Rv, runit = _values_and_unit(Rho, coord_unit)
    Pv, _ = _values_and_unit(Phi, units.rad)

    # Ensure 2D grid shapes and facecolors averaged to cell colors (M-1, N-1, 4)
    def _as_2d(a):
        a = np.asarray(a)
        if a.ndim == 1:
            return a
        return a

    R_arr = _as_2d(Rv)
    P_arr = _as_2d(Pv)

    # Determine grid shape
    if R_arr.ndim == 2 and P_arr.ndim == 2:
        Nr, Np_ = R_arr.shape
        if rgba.shape[0] != Nr or rgba.shape[1] != Np_:
            raise ValueError("field shape must match Rho/Phi grid shape.")
        # Average to cell colors
        if Nr > 1 and Np_ > 1:
            fc = 0.25*(rgba[:-1, :-1, :] + rgba[1:, :-1, :] + rgba[:-1, 1:, :] + rgba[1:, 1:, :])
        else:
            fc = rgba
        # Create a dummy C to satisfy pcolormesh API, then set facecolors explicitly
        qm = ax.pcolormesh(
            P_arr, R_arr, np.zeros((max(Nr-1,1), max(Np_-1,1))),
            shading='flat', antialiased=False, edgecolors='none'
        )
        # Detach scalar array so our facecolors are respected
        qm.set_array(None)
        qm.set_cmap(None)
        qm.set_facecolors(fc.reshape(-1, fc.shape[-1]))
        # Rasterize heavy quadmesh to accelerate vector exports
        try:
            qm.set_rasterized(True)
        except Exception:
            pass
    else:
        # Assume 1D vectors for radii and angles
        r = np.asarray(Rv).ravel()
        p = np.asarray(Pv).ravel()
        Nr = r.size; Np_ = p.size
        if rgba.shape[0] != Nr or rgba.shape[1] != Np_:
            raise ValueError("For 1D Rho/Phi, field must have shape (len(Rho), len(Phi)).")
        if Nr > 1 and Np_ > 1:
            fc = 0.25*(rgba[:-1, :-1, :] + rgba[1:, :-1, :] + rgba[:-1, 1:, :] + rgba[1:, 1:, :])
            qm = ax.pcolormesh(p, r, np.zeros((Nr-1, Np_-1)), shading='flat', antialiased=False, edgecolors='none')
            qm.set_array(None)
            qm.set_cmap(None)
            qm.set_facecolors(fc.reshape(-1, fc.shape[-1]))
            # Rasterize the heavy quadmesh for faster vector export
            try:
                qm.set_rasterized(True)
            except Exception:
                pass
        else:
            # Fallback: single cell
            qm = ax.pcolormesh(p, r, np.zeros((1, 1)), shading='flat', antialiased=False, edgecolors='none')
            qm.set_array(None)
            qm.set_cmap(None)
            qm.set_facecolors(rgba.reshape(-1, rgba.shape[-1]))
            try:
                qm.set_rasterized(True)
            except Exception:
                pass

    # Labels and title
    if rlabel is None:
        rlabel = _axis_label_from_unit("r", runit)
    # ax.set_ylabel(rlabel)
    if title:
        ax.set_title(title)
    rmax = float(np.nanmax(Rv))
    ax.set_ylim(0, rmax)

    # Radial ticks at k·a if core_radius is provided (in coord_unit)
    if core_radius is not None:
        try:
            a_vals, _ = _values_and_unit(core_radius, coord_unit)
            a = float(np.asarray(a_vals).squeeze())
            if np.isfinite(a) and a > 0:
                kmax = int(np.floor(rmax / a))
                if kmax >= 1:
                    ticks = a * np.arange(1, kmax + 1)
                    ax.set_rticks(ticks)

        except Exception:
            # ignore conversion errors and keep default ticks
            pass

    # Grid control
    if show_grid:
        gk = {"linestyle": ":", "linewidth": 1.0, "alpha": 0.4, "color": "white"}
        if grid_kwargs:
            gk.update(grid_kwargs)
        ax.grid(True, **gk)
    else:
        ax.grid(False)

    # Hide radial tick labels (keep ticks and gridlines)
    try:
        ax.set_yticklabels([])
        for lab in ax.yaxis.get_ticklabels():
            lab.set_visible(False)
        # Also hide any offset text if present
        ax.yaxis.get_offset_text().set_visible(False)
    except Exception:
        pass

    # Colorbars (separate mappables)
    cb_phase = cb_mag = None
    if show_phase_cbar:
        style = str(phase_cbar_style).lower()
        if style == "wheel":
            # Draw a circular HSV phase wheel as an inset
            x0, y0, w, h = phase_wheel_pos
            axw = ax.inset_axes([x0, y0, w, h], transform=ax.transAxes)
            N = int(phase_wheel_res)
            yy, xx = np.ogrid[-1:1:N*1j, -1:1:N*1j]
            rr = np.sqrt(xx*xx + yy*yy)
            ang = np.arctan2(yy, xx)
            hue_w = np.mod(ang / (2*np.pi), 1.0)
            s_w = np.ones_like(hue_w)
            v_w = np.ones_like(hue_w)
            rgb_w = hsv_to_rgb(np.stack([hue_w, s_w, v_w], axis=-1))
            alpha_w = ((.5 <= rr) & (rr <= 0.9)).astype(float)
            rgba_w = np.dstack([rgb_w, alpha_w])
            axw.imshow(rgba_w, origin='lower', extent=[-1, 1, -1, 1])
            axw.set_xticks([]); axw.set_yticks([])
            axw.set_aspect('equal'); 
            # axw.set_title(phase_label, fontsize=9, pad=2)
            #hide axis
            axw.axis('off')
            # Keep cb_phase as None for wheel style
            cb_phase = None
        elif style == "edge":
            # Draw a thin HSV ring hugging the outer edge of the polar axis
            axw = ax.inset_axes([0, 0, 1, 1], transform=ax.transAxes, projection='polar', zorder=9)
            axw.set_facecolor('none')
            axw.set_frame_on(False)
            axw.grid(False)
            axw.set_clip_on(False)
            # Match orientation with main axis if possible
            try:
                direction = ax.get_theta_direction()
                axw.set_theta_direction(direction)
            except Exception:
                pass
            try:
                zero = ax.get_theta_zero_location()
                axw.set_theta_zero_location(zero)
            except Exception:
                try:
                    axw.set_theta_zero_location('E')
                except Exception:
                    pass
            # Normalized ring coordinates
            w = float(np.clip(phase_edge_width, 1e-3, 0.5))
            gap = float(np.clip(phase_edge_gap, 0.0, 0.2))
            r0 = max(0.0, 1.0 - gap - w)
            r1 = max(r0 + 1e-3, 1.0 - gap)
            # Use 2D mesh with Gouraud shading for smooth angular gradients
            Nt = int(max(72, phase_edge_theta_N)) + 1  # +1 to close the seam at ±π
            theta = np.linspace(-np.pi, np.pi, Nt)
            r = np.array([r0, r1])  # just two radial samples define a ring strip
            TT, RR = np.meshgrid(theta, r, indexing='xy')  # shapes (2, Nt)
            # Color by phase (theta) using a cyclic HSV colormap
            phase_norm = Normalize(vmin=-np.pi, vmax=np.pi)
            cmap_phase = _rolled_cmap(phase_cmap, shift=-0.5)
            pc = axw.pcolormesh(
                TT, RR, TT,  # C same shape as coordinates for Gouraud
                shading='gouraud', cmap=cmap_phase, norm=phase_norm,
                edgecolors='none', antialiased=True
            )
            try:
                pc.set_rasterized(True)
            except Exception:
                pass
            axw.set_rlim(0.0, 1.0)
            axw.set_rticks([])
            axw.set_thetagrids([])
            cb_phase = None
        else:
            phase_norm = Normalize(vmin=-np.pi, vmax=np.pi)
            cmap_phase = _rolled_cmap(phase_cmap, shift=-0.5)
            sm_phase = plt.cm.ScalarMappable(norm=phase_norm, cmap=cmap_phase)
            sm_phase.set_array([])
            cb_phase = fig.colorbar(sm_phase, ax=ax, pad=cbar_pad, shrink=cbar_shrink, fraction=cbar_fraction)
            cb_phase.set_label(phase_label)
            cb_phase.set_ticks(list(phase_ticks))
            if tuple(phase_ticks) == (-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi):
                cb_phase.set_ticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])

    if show_mag_cbar:
        mag_norm = Normalize(vmin=0.0, vmax=vmax_eff)
        sm_mag = plt.cm.ScalarMappable(norm=mag_norm, cmap=mag_cmap)
        sm_mag.set_array([])
        cb_mag = fig.colorbar(sm_mag, ax=ax, pad=cbar_pad + 0.02, shrink=cbar_shrink, fraction=cbar_fraction)
        cb_mag.set_label(mag_label)
        cb_mag.formatter.set_powerlimits((0, 0))
        cb_mag.update_ticks()

    if created_fig:
        plt.show()

    return fig, ax, cb_phase, cb_mag


def animate_fields_xy(
    *,
    # --- Option A: give modes (one or many) ---
    modes=None,            # GuidedMode or list[GuidedMode]
    weights=None,          # complex or list[complex] (amplitudes/relative phases), default 1
    n_radii=2.0,           # grid half-size in units of core radius (when building grid)
    Np=200,                # grid resolution per axis

    # --- Option B: give fields with their own ω ---
    fields=None,           # list of tuples (E, H, omega) with E/H phasors on same X,Y grid
    X=None, Y=None,        # grid for Option B (required if fields given)

    # --- Plot controls ---
    show=("E", "H"),       # any subset of {"E","H"}
    scale="auto",          # arrow size multiplier (1.0 if "auto")
    zscale=None,           # color scale for Ez/Hz (robust auto if None)
    cmap="RdBu_r",
    n_frames=60,
    interval=50,
    figsize=(8, 4.5),

    # --- Stability knobs ---
    robust_p=99.0,         # percentile for robust scaling of color + field normalization
    quiver_density=20,
    quiver_scale=25.0,     # base quiver scale (smaller -> longer arrows)
    eps=1e-30,
):
    """
    Animate instantaneous fields in an XY cross-section.

    Option 2: RMS normalization for quiver arrows.
      • Arrows preserve relative magnitudes.
      • Transverse arrows are divided by RMS(|F_perp|) (computed once at theta=0),
        giving a more energy-like visual normalization than max/percentile.
      • z-colors use robust percentile scaling (unless zscale given).
      • Works with astropy units via global _HAS_UNITS / units.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from scipy.constants import c as c0

    # ---------------------------
    # Unit helpers using your global _HAS_UNITS/units
    # ---------------------------
    def _to_float_m(x):
        if _HAS_UNITS and hasattr(x, "unit"):
            return float(x.to_value(units.m))
        return float(x)

    def _to_um(arr):
        if _HAS_UNITS and hasattr(arr, "unit"):
            return arr.to_value(units.um)
        return np.asarray(arr) * 1e6  # assume meters

    # ---------------------------
    # Robust scaling helpers
    # ---------------------------
    def _robust_scale(arr, p=99.0, eps_=1e-30):
        a = np.asarray(arr)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return 1.0
        s = np.percentile(np.abs(a), p)
        if (not np.isfinite(s)) or (s <= eps_):
            return 1.0
        return float(s)

    def _robust_vec_scale(Fx, Fy, p=99.0, eps_=1e-30):
        mag = np.sqrt(np.asarray(Fx)**2 + np.asarray(Fy)**2)
        return _robust_scale(mag, p=p, eps_=eps_)

    def _rms_perp_scale(Fx, Fy, eps_=1e-30):
        """
        RMS(|F_perp|) over finite entries.
        Returns sqrt(mean(Fx^2 + Fy^2)).
        """
        Fx = np.asarray(Fx)
        Fy = np.asarray(Fy)
        mag2 = Fx**2 + Fy**2
        mag2 = mag2[np.isfinite(mag2)]
        if mag2.size == 0:
            return 1.0
        s = float(np.sqrt(np.mean(mag2)))
        if (not np.isfinite(s)) or (s <= eps_):
            return 1.0
        return s

    # ---------------------------
    # Build component list (E_k, H_k, ω_k)
    # ---------------------------
    comps = []  # list of (E_k, H_k, omega_k)

    if modes is not None:
        modes = modes if isinstance(modes, (list, tuple)) else [modes]
        if weights is None:
            weights = [1.0] * len(modes)
        if not isinstance(weights, (list, tuple)):
            weights = [weights]
        if len(weights) != len(modes):
            raise ValueError("weights must match modes in length.")

        # Build grid if X,Y not given yet (use first mode's fibre radius)
        if X is None or Y is None:
            a = modes[0].fibre.core_radius  # meters or Quantity
            a_m = _to_float_m(a)
            L = float(n_radii) * a_m
            x = np.linspace(-L, L, Np)
            y = np.linspace(-L, L, Np)
            X, Y = np.meshgrid(x, y)

        # Evaluate each mode on the same grid
        for m, w in zip(modes, weights):
            E = np.asarray(m.E(x=X, y=Y)) * w
            H = np.asarray(m.H(x=X, y=Y)) * w

            wl = m.wavelength  # maybe Quantity (e.g. nm)
            wl_m = float(wl.to_value(units.m)) if (_HAS_UNITS and hasattr(wl, "unit")) else float(wl)
            omega = 2.0 * np.pi * c0 / wl_m
            comps.append((E, H, omega))

    if fields is not None:
        if X is None or Y is None:
            raise ValueError("When using 'fields=', you must also provide X and Y.")
        for (E, H, omega) in fields:
            comps.append((np.asarray(E), np.asarray(H), float(omega)))

    if not comps:
        raise ValueError("Provide either 'modes=' or 'fields='.")

    # ---------------------------
    # Ensure X,Y are numeric arrays in meters
    # ---------------------------
    if _HAS_UNITS and hasattr(X, "unit"):
        X = X.to_value(units.m)
    if _HAS_UNITS and hasattr(Y, "unit"):
        Y = Y.to_value(units.m)

    X = np.asarray(X)
    Y = np.asarray(Y)

    X_um = _to_um(X)
    Y_um = _to_um(Y)
    extent = [X_um.min(), X_um.max(), Y_um.min(), Y_um.max()]

    # Reference frequency for phase driving
    omegas = np.array([om for (_, _, om) in comps], dtype=float)
    omega_ref = omegas[0]
    ratios = omegas / omega_ref

    showE = "E" in show
    showH = "H" in show
    ncols = int(showE) + int(showH)
    if ncols == 0:
        raise ValueError("Nothing to show: set show=('E',), ('H',) or ('E','H').")

    # ---------------------------
    # Summed instantaneous snapshot
    # ---------------------------
    def _sum_snap(theta, Escale=1.0, Hscale=1.0):
        Et = 0.0
        Ht = 0.0
        for (Ek, Hk, rk) in zip((c[0] for c in comps), (c[1] for c in comps), ratios):
            ph = np.exp(-1j * rk * theta)
            Et = Et + Ek * ph
            Ht = Ht + Hk * ph

        Ereal = np.real(Et) * Escale
        Hreal = np.real(Ht) * Hscale

        if Ereal.shape[-1] != 3 or Hreal.shape[-1] != 3:
            raise ValueError("Fields must have last axis of length 3 (Fx,Fy,Fz).")

        Ex, Ey, Ez = Ereal[..., 0], Ereal[..., 1], Ereal[..., 2]
        Hx, Hy, Hz = Hreal[..., 0], Hreal[..., 1], Hreal[..., 2]
        return (Ex, Ey, Ez), (Hx, Hy, Hz)

    # First pass: choose robust plot normalization factors
    (Ex_raw, Ey_raw, Ez_raw), (Hx_raw, Hy_raw, Hz_raw) = _sum_snap(0.0, Escale=1.0, Hscale=1.0)

    E0_scale = max(
        _robust_vec_scale(Ex_raw, Ey_raw, p=robust_p, eps_=eps),
        _robust_scale(Ez_raw, p=robust_p, eps_=eps),
    )
    H0_scale = max(
        _robust_vec_scale(Hx_raw, Hy_raw, p=robust_p, eps_=eps),
        _robust_scale(Hz_raw, p=robust_p, eps_=eps),
    )

    Escale = 1.0 / (E0_scale if E0_scale > eps else 1.0)
    Hscale = 1.0 / (H0_scale if H0_scale > eps else 1.0)

    # Normalized snapshot for plotting
    (Ex0, Ey0, Ez0), (Hx0, Hy0, Hz0) = _sum_snap(0.0, Escale=Escale, Hscale=Hscale)

    # Robust color scaling for z components
    if zscale is None:
        zvmax = 0.0
        if showE:
            zvmax = max(zvmax, _robust_scale(Ez0, p=robust_p, eps_=eps))
        if showH:
            zvmax = max(zvmax, _robust_scale(Hz0, p=robust_p, eps_=eps))
        if (not np.isfinite(zvmax)) or (zvmax <= eps):
            zvmax = 1.0
    else:
        zvmax = float(zscale)

    # Downsample for quiver
    Ny, Nx = X.shape
    stride = max(1, int(min(Nx, Ny) // max(1, int(quiver_density))))
    Xs = X_um[::stride, ::stride]
    Ys = Y_um[::stride, ::stride]

    # Arrow size multiplier
    if (scale == "auto") or (scale is None):
        scale_factor = 1.0
    else:
        try:
            scale_factor = float(scale)
        except Exception:
            scale_factor = 1.0

    # Initial downsampled transverse fields
    Exq0 = Ex0[::stride, ::stride]
    Eyq0 = Ey0[::stride, ::stride]
    Hxq0 = Hx0[::stride, ::stride]
    Hyq0 = Hy0[::stride, ::stride]

    # RMS normalization scales (FIXED across animation; avoids "breathing")
    sE = 4*_rms_perp_scale(Exq0, Eyq0, eps_=eps)
    sH = 4*_rms_perp_scale(Hxq0, Hyq0, eps_=eps)

    # Normalized initial quiver components
    Exn0 = Exq0 / sE
    Eyn0 = Eyq0 / sE
    Hxn0 = Hxq0 / sH
    Hyn0 = Hyq0 / sH

    # Matplotlib quiver: arrow length ~ U / scale
    qscale_E = quiver_scale / max(scale_factor, eps)
    qscale_H = quiver_scale / max(scale_factor, eps)

    # ---------------------------
    # Figure and artists
    # ---------------------------
    fig, axes = plt.subplots(1, ncols, figsize=figsize, sharex=True, sharey=True)
    if ncols == 1:
        axes = [axes]
    artists = []

    idx = 0
    if showE:
        ax = axes[idx]; idx += 1
        imE = ax.imshow(
            Ez0.T, extent=extent, origin="lower", cmap=cmap,
            vmin=-zvmax, vmax=zvmax, interpolation="nearest", aspect="equal"
        )
        qE = ax.quiver(
            Xs, Ys, Exn0, Eyn0,
            scale=qscale_E, color="k", pivot="middle", linewidth=0.5
        )
        ax.set_title("Electric field")
        artists += [imE, qE]

    if showH:
        ax = axes[idx]; idx += 1
        imH = ax.imshow(
            Hz0.T, extent=extent, origin="lower", cmap=cmap,
            vmin=-zvmax, vmax=zvmax, interpolation="nearest", aspect="equal"
        )
        qH = ax.quiver(
            Xs, Ys, Hxn0, Hyn0,
            scale=qscale_H, color="k", pivot="middle", linewidth=0.5
        )
        ax.set_title("Magnetic field")
        artists += [imH, qH]

    for ax in axes:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$ [$\mu$m]")
    axes[0].set_ylabel(r"$y$ [$\mu$m]")
    plt.tight_layout()

    # ---------------------------
    # Animation update
    # ---------------------------
    def update(i):
        theta = 2.0 * np.pi * i / n_frames
        (Ex, Ey, Ez), (Hx, Hy, Hz) = _sum_snap(theta, Escale=Escale, Hscale=Hscale)

        aidx = 0
        if showE:
            artists[aidx].set_data(Ez.T); aidx += 1
            Exq = Ex[::stride, ::stride]
            Eyq = Ey[::stride, ::stride]
            artists[aidx].set_UVC(Exq / sE, Eyq / sE); aidx += 1

        if showH:
            artists[aidx].set_data(Hz.T); aidx += 1
            Hxq = Hx[::stride, ::stride]
            Hyq = Hy[::stride, ::stride]
            artists[aidx].set_UVC(Hxq / sH, Hyq / sH); aidx += 1

        return artists

    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)
    plt.close(fig)
    return anim
