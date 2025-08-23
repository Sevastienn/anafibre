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
    b_vals = np.linspace(0.0, 1.0, Npoints)
    VV, BB = np.meshgrid(V_vals, b_vals)

    F_grid = F_dispersion(fibre, ell=ell, b=BB, V=VV, mode_type=mode_type)

    absmax = np.nanmax(np.abs(F_grid[np.isfinite(F_grid)]))
    if np.isnan(absmax):
        # Choose defaults or skip the plot
        vmin, vmax = None, None
    else:
        vmax = absmax
        vmin = -vmax
    # print(f"Max F: {vmax}, Min F: {vmin}")
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


def plot_xy_vector_field(
    X, Y, F, ax=None, scale=40, zscale=None, cmap='RdBu_r', stride=4, title=None,
    xlabel=None, ylabel=None, colorbar_label=None, colorbar=False,
    density=1.0, linewidth=1, color='k', type='quiver', labels=False
):
    """
    Plot XY vector field and colormap, showing both inside and outside fibre.
    X, Y : 2D arrays (SI units or astropy Quantity)
    F    : (..., 3) vector field (real or Quantity)
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
        ax.quiver(
            Xv[::stride, ::stride], Yv[::stride, ::stride],
            Fxv[::stride, ::stride], Fyv[::stride, ::stride],
            scale=scale, color='black', pivot='middle', linewidth=0.5, 
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

# def plot_polar_vector_field(
#     Rho, Phi, F, ax=None, scale=20, stride_r=1, stride_phi=3,
#     title=None, rlabel=None, philabel=None, colorbar_label=None, colorbar=False, cmap='RdBu_r'
# ):
#     """
#     Plot vector field (Fr, Fphi, Fz) on polar axes.
    
#     Rho, Phi : 2D arrays [radial, angular] (astropy Quantity or floats)
#     F        : (..., 3) vector field in cylindrical components (Fr, Fphi, Fz)
#     ax       : matplotlib polar axes (if None, a new one is created)
#     """

#     import numpy as np
#     import matplotlib.pyplot as plt

#     # Handle units for axis labels
#     def _get_unit_and_val(arr):
#         if hasattr(arr, 'unit'):
#             return arr.unit, arr.value
#         return None, arr

#     r_unit, Rv = _get_unit_and_val(Rho)
#     phi_unit, Phiv = _get_unit_and_val(Phi)

#     Fx, Fy, Fz = F[..., 0], F[..., 1], F[..., 2]
#     Fxv = Fx.real.value if hasattr(Fx, 'unit') else Fx.real
#     Fyv = Fy.real.value if hasattr(Fy, 'unit') else Fy.real
#     Fzv = Fz.real.value if hasattr(Fz, 'unit') else Fz.real

#     if ax is None:
#         fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5, 5))
#     else:
#         fig = ax.figure

#     # Colormap for Fz (radial, angular) on polar axes (optional)
#     absmax = np.max(np.abs(Fzv))
#     if absmax != 0:
#         c = ax.pcolormesh(Phiv, Rv, Fzv, shading='auto', cmap=cmap, vmin=-absmax, vmax=absmax)
#         if colorbar:
#             cb_label = colorbar_label or r"$F_z$"
#             plt.colorbar(c, ax=ax, label=cb_label, pad=0.1, shrink=0.7)

#     # Plot vector field: Fx, Fy as arrows
#     ax.quiver(
#         Phiv[::stride_r, ::stride_phi], Rv[::stride_r, ::stride_phi],
#         Fxv[::stride_r, ::stride_phi], Fyv[::stride_r, ::stride_phi],
#         pivot='middle', color='black', scale=scale, width=0.006,
#     )

#     # Labels and ticks
#     r_unit_str = f"[{r_unit:latex_inline}]" if r_unit else "[m]"
#     ax.set_title(title or "")
#     if rlabel is not False:
#         ax.set_ylabel(rlabel or f"r {r_unit_str}")
#     ax.set_rticks([])      # Hide radial ticks (optional)
#     ax.set_xticks([])      # Hide phi ticks (optional)
#     ax.set_yticklabels([])
#     ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
#     return ax

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



# def animate_fields_xy(mode=None, X=None, Y=None, E=None, H=None, *,
#                       n_radii=2, Np=200,
#                       scale=40, zscale=None,
#                       n_frames=60, interval=50,
#                       figsize=(10, 5), cmap='RdBu_r'):
#     """
#     Animate instantaneous E and H fields in XY cross-section.

#     Two usage modes:
#     ----------------
#     1. Provide E, H, X, Y directly (for superpositions or precomputed fields).
#     2. Provide a GuidedMode `mode` (and optionally `n_radii`, `Np`);
#        the grid is generated automatically.

#     Parameters
#     ----------
#     mode : GuidedMode, optional
#         Guided mode to animate. If given, X/Y/E/H can be omitted.
#     X, Y : 2D arrays, optional
#         Grid (with or without astropy units). Required if E,H are given directly.
#     E, H : 3D arrays, optional
#         Complex phasor fields (...,3). Required if not using `mode`.
#     n_radii : float, optional
#         Grid half-size in units of fibre core radius (default 2).
#     Np : int, optional
#         Grid resolution per axis (default 200).
#     scale : float, optional
#         Scaling for quiver arrows.
#     zscale : float, optional
#         Colormap scale for Ez/Hz. Auto if None.
#     n_frames : int, optional
#         Number of animation frames per oscillation cycle.
#     interval : int, optional
#         Delay between frames in ms.
#     figsize : tuple, optional
#         Figure size.
#     cmap : str, optional
#         Colormap for Ez/Hz.

#     Returns
#     -------
#     anim : matplotlib.animation.FuncAnimation
#         Animation object. Use `anim.to_jshtml()` in Jupyter or `anim.save("out.mp4")`.
#     """

#     import matplotlib.pyplot as plt
#     from matplotlib.animation import FuncAnimation
#     from scipy.constants import epsilon_0 as eps0, mu_0 as mu0, c as c0

#     # --------------------
#     # Case 2: build grid + fields from mode
#     # --------------------
#     if mode is not None and (E is None or H is None):
#         # Generate grid
#         a = mode.fibre.core_radius
#         L = (n_radii * a )
#         x = np.linspace(-L, L, Np)
#         y = np.linspace(-L, L, Np)
#         X, Y = np.meshgrid(x, y)

#         # Compute fields
#         E = mode.E(x=X, y=Y)
#         H = mode.H(x=X, y=Y)
#         X, Y = X*units.m, Y*units.m

#     # --------------------
#     # Case 1: all provided directly
#     # --------------------
#     if E is None or H is None or X is None or Y is None:
#         raise ValueError("Provide either (mode) OR (E, H, X, Y).")

#     # Frequency (only needed for title)
#     omega = None
#     if mode is not None:
#         omega = 2*np.pi * c0 / mode.wavelength.to_value(units.m)

#     # Grid size
#     Np = X.shape[0]
#     stride = max(1, Np // 20)

#     # Downsampled grid for quiver
#     Xs = X[::stride, ::stride]
#     Ys = Y[::stride, ::stride]
#     if hasattr(Xs, "unit"):
#         Xs = Xs.to_value(units.um)
#         Ys = Ys.to_value(units.um)

#     # Convert full grids for imshow extents
#     Xv = X.to_value(units.um) if hasattr(X, "unit") else X
#     Yv = Y.to_value(units.um) if hasattr(Y, "unit") else Y
#     extent = [Xv.min(), Xv.max(), Yv.min(), Yv.max()]

#     # Snapshot helper
#     def snapshot(F, theta, scale_factor=1.0):
#         Fθ = np.real(F * np.exp(-1j*theta)) * scale_factor
#         return Fθ[...,0], Fθ[...,1], Fθ[...,2]

#     Escale = np.sqrt(eps0)
#     Hscale = np.sqrt(mu0)

#     Ex0, Ey0, Ez0 = snapshot(E, 0.0, Escale)
#     Hx0, Hy0, Hz0 = snapshot(H, 0.0, Hscale)

#     # Auto zscale
#     if zscale is None:
#         zvmax = max(np.max(np.abs(Ez0)), np.max(np.abs(Hz0)))
#     else:
#         zvmax = zscale

#     # Build figure
#     fig, (axE, axH) = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
#     imE = axE.imshow(Ez0.T, extent=extent, origin='lower', cmap=cmap,
#                      vmin=-zvmax, vmax=zvmax, interpolation='nearest', aspect='equal')
#     imH = axH.imshow(Hz0.T, extent=extent, origin='lower', cmap=cmap,
#                      vmin=-zvmax, vmax=zvmax, interpolation='nearest', aspect='equal')

#     qE = axE.quiver(Xs, Ys, Ex0[::stride, ::stride], Ey0[::stride, ::stride],
#                     scale=scale, color='k', pivot='middle', linewidth=0.5)
#     qH = axH.quiver(Xs, Ys, Hx0[::stride, ::stride], Hy0[::stride, ::stride],
#                     scale=scale, color='k', pivot='middle', linewidth=0.5)

#     for ax, title in zip((axE, axH),
#                          (r'Electric field  $\Re\{\mathbf{E}e^{-i\omega t}\}$',
#                           r'Magnetic field  $\Re\{\mathbf{H}e^{-i\omega t}\}$')):
#         ax.set_xlim(extent[0], extent[1])
#         ax.set_ylim(extent[2], extent[3])
#         ax.set_aspect('equal')
#         ax.set_xlabel('x [$\mu$m]')
#         ax.set_title(title)
#     axE.set_ylabel('y [$\mu$m]')
#     # plt.tight_layout()

#     # Animation callback
#     def update(i):
#         theta = 2*np.pi * i / n_frames
#         Ex, Ey, Ez = snapshot(E, theta, Escale)
#         Hx, Hy, Hz = snapshot(H, theta, Hscale)

#         imE.set_data(Ez.T)
#         imH.set_data(Hz.T)

#         qE.set_UVC(Ex[::stride, ::stride], Ey[::stride, ::stride])
#         qH.set_UVC(Hx[::stride, ::stride], Hy[::stride, ::stride])

#         return imE, imH, qE, qH

#     anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)
#     plt.close(fig)
#     return anim


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
    show=("E","H"),        # any subset of {"E","H"}; e.g. ("E",) to animate only E
    scale=40,              # quiver scale
    zscale=None,           # color scale for Ez/Hz (auto if None)
    cmap='RdBu_r',
    n_frames=60,           # frames over θ∈[0,2π)
    interval=50,           # ms between frames
    figsize=(8,4.5),
):
    """
    Animate instantaneous fields in an XY cross‑section.

    Usage:
      • Single or multiple modes (possibly different wavelengths):
          anim = animate_fields_xy(modes=[m1, m2], weights=[1.0, 0.7*np.exp(1j*np.pi/3)])
          # grid auto‑generated to ±n_radii * core_radius of the FIRST mode

      • Precomputed fields with distinct frequencies on a given grid:
          fields=[(E1,H1,omega1), (E2,H2,omega2)], X=..., Y=...
          anim = animate_fields_xy(fields=fields, X=X, Y=Y)

    Returns: matplotlib.animation.FuncAnimation
    """
    from matplotlib.animation import FuncAnimation
    from scipy.constants import epsilon_0 as eps0, mu_0 as mu0, c as c0

    def _to_um(arr):
        if _HAS_UNITS and hasattr(arr, "unit"):
            return arr.to_value(units.um)
        return np.asarray(arr) * 1e6  # assume meters

    # ---------------------------
    # Build component list (E_k, H_k, ω_k)
    # ---------------------------
    comps = []  # list of (E_k, H_k, omega_k)

    if modes is not None:
        # Normalize to list
        modes = modes if isinstance(modes, (list, tuple)) else [modes]
        if weights is None:
            weights = [1.0] * len(modes)
        if not isinstance(weights, (list, tuple)):
            weights = [weights]
        if len(weights) != len(modes):
            raise ValueError("weights must match modes in length.")

        # Build grid if X,Y not given yet (use first mode's fibre radius)
        if X is None or Y is None:
            a = modes[0].fibre.core_radius  # meters
            L = float(n_radii) * a
            x = np.linspace(-L, L, Np)
            y = np.linspace(-L, L, Np)
            X, Y = np.meshgrid(x, y)

        # Evaluate each mode on the same grid
        for m, w in zip(modes, weights):
            E = np.asarray(m.E(x=X, y=Y)) * w
            H = np.asarray(m.H(x=X, y=Y)) * w
            omega = 2*np.pi * c0 / np.asarray(m.wavelength if not _HAS_UNITS else m.wavelength.to_value(units.m))
            comps.append((E, H, omega))

    if fields is not None:
        if X is None or Y is None:
            raise ValueError("When using 'fields=', you must also provide X and Y.")
        for (E, H, omega) in fields:
            comps.append((np.asarray(E), np.asarray(H), float(omega)))

    if not comps:
        raise ValueError("Provide either 'modes=' or 'fields='.")

    # Ensure X,Y are numpy arrays (meters); convert copies for plotting in μm
    X = np.asarray(X); Y = np.asarray(Y)
    X_um = _to_um(X); Y_um = _to_um(Y)
    extent = [X_um.min(), X_um.max(), Y_um.min(), Y_um.max()]

    # Reference frequency for phase driving
    omegas = np.array([om for (_,_,om) in comps], dtype=float)
    omega_ref = omegas[0]
    ratios = omegas / omega_ref  # r_k

    # Precompute scales for instantaneous snapshots
    Escale = np.sqrt(eps0)
    Hscale = np.sqrt(mu0)

    # Convenience: what panels to draw
    showE = "E" in show
    showH = "H" in show
    ncols = int(showE) + int(showH)
    if ncols == 0:
        raise ValueError("Nothing to show: set show=('E',), ('H',) or ('E','H').")

    # Initial snapshot (θ=0)
    def _sum_snap(theta):
        # Sum_k F_k * exp(-i r_k theta)
        Et = 0; Ht = 0
        for (Ek, Hk, rk) in zip((c[0] for c in comps), (c[1] for c in comps), ratios):
            ph = np.exp(-1j * rk * theta)
            Et = Et + Ek * ph
            Ht = Ht + Hk * ph

        # Real instantaneous fields; split last axis (…, 3) → (Ex, Ey, Ez)
        Ereal = np.real(Et) * Escale
        Hreal = np.real(Ht) * Hscale

        # Safety: ensure vector last axis exists and is length 3
        if Ereal.shape[-1] != 3 or Hreal.shape[-1] != 3:
            raise ValueError("Fields must have last axis of length 3 (Fx,Fy,Fz).")

        Ex, Ey, Ez = Ereal[..., 0], Ereal[..., 1], Ereal[..., 2]
        Hx, Hy, Hz = Hreal[..., 0], Hreal[..., 1], Hreal[..., 2]
        return (Ex, Ey, Ez), (Hx, Hy, Hz)

    # Initial fields at theta=0
    (Ex0, Ey0, Ez0), (Hx0, Hy0, Hz0) = _sum_snap(0.0)

    # z-scale
    if zscale is None:
        zvmax = 0.0
        if showE: zvmax = max(zvmax, np.max(np.abs(Ez0)))
        if showH: zvmax = max(zvmax, np.max(np.abs(Hz0)))
        if zvmax == 0: 
            zvmax = 1.0
    else:
        zvmax = zscale

    # Downsample for quiver
    Np_guess = X.shape[0]
    stride = max(1, Np_guess // 20)
    Xs = X_um[::stride, ::stride]
    Ys = Y_um[::stride, ::stride]

    # Figure and artists
    fig, axes = plt.subplots(1, ncols, figsize=figsize, sharex=True, sharey=True)
    if ncols == 1: axes = [axes]
    artists = []

    idx = 0
    if showE:
        ax = axes[idx]; idx += 1
        imE = ax.imshow(Ez0.T, extent=extent, origin='lower', cmap=cmap,
                        vmin=-zvmax, vmax=zvmax, interpolation='nearest', aspect='equal')
        qE = ax.quiver(Xs, Ys, Ex0[::stride, ::stride], Ey0[::stride, ::stride],
                       scale=scale, color='k', pivot='middle', linewidth=0.5)
        ax.set_title(r'Electric field')
        artists += [imE, qE]

    if showH:
        ax = axes[idx]; idx += 1
        imH = ax.imshow(Hz0.T, extent=extent, origin='lower', cmap=cmap,
                        vmin=-zvmax, vmax=zvmax, interpolation='nearest', aspect='equal')
        qH = ax.quiver(Xs, Ys, Hx0[::stride, ::stride], Hy0[::stride, ::stride],
                       scale=scale, color='k', pivot='middle', linewidth=0.5)
        ax.set_title(r'Magnetic field')
        artists += [imH, qH]

    for ax in axes:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect('equal')
        ax.set_xlabel('$x$ [$\\mu$m]')
    axes[0].set_ylabel('$y$ [$\\mu$m]')
    plt.tight_layout()

    # Animation callback stepping θ∈[0,2π)
    def update(i):
        theta = 2*np.pi * i / n_frames
        (Ex, Ey, Ez), (Hx, Hy, Hz) = _sum_snap(theta)

        aidx = 0
        if showE:
            artists[aidx].set_data(Ez.T); aidx += 1
            artists[aidx].set_UVC(Ex[::stride, ::stride], Ey[::stride, ::stride]); aidx += 1
        if showH:
            artists[aidx].set_data(Hz.T); aidx += 1
            artists[aidx].set_UVC(Hx[::stride, ::stride], Hy[::stride, ::stride]); aidx += 1
        return artists

    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)
    plt.close(fig)
    return anim