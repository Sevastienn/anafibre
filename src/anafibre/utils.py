"""
Utility functions and optional dependency management for Anafibre.
Includes unit handling and color conversion for wavelength visualisation.
"""
try:
    import astropy.units as units
    _HAS_UNITS = True
except ImportError:
    _HAS_UNITS = False
    units = None  # Gracefully handle absence of astropy

try:
    from refractiveindex.refractiveindex import RefractiveIndexMaterial as RIMaterial, NoExtinctionCoefficient
    _HAS_REFRACTIVEINDEX = True
except ImportError:
    _HAS_REFRACTIVEINDEX = False
    RIMaterial = None
    NoExtinctionCoefficient = Exception

def _strip_unit(val, unit=None):
    """Return the numeric value of an input with optional unit conversion.

    Parameters
    ----------
    val : Any
        Scalar or array-like value. If ``astropy`` is available and ``val`` has a
        ``unit`` attribute, the value is converted to plain numbers.
    unit : astropy.units.Unit | None, optional
        Target unit for conversion when ``val`` is a quantity. If ``None``,
        SI base units are used.

    Returns
    -------
    Any
        Numeric value (or array of values) when ``val`` is a quantity; otherwise
        returns ``val`` unchanged.
    """
    if _HAS_UNITS and hasattr(val, 'unit'):
        if unit is not None:
            return val.to(unit).value
        return val.si.value
    return val

def wavelength_to_rgb(wl):
    """Map visible wavelength to an approximate sRGB color.

    Parameters
    ----------
    wl : float
        Wavelength in nanometers.

    Returns
    -------
    tuple[float, float, float]
        Normalised ``(R, G, B)`` values in the range ``[0, 1]``.
        Values outside the visible range (380-780 nm) return black ``(0, 0, 0)``.

    Notes
    -----
    Uses a standard piecewise approximation for visible-spectrum mapping rather
    than a colorimetrically exact CIE conversion.
    """
    if wl < 380 or wl > 780:
        return (0, 0, 0)  # Outside visible range

    if wl < 440:
        R = -(wl - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif wl < 490:
        R = 0.0
        G = (wl - 440) / (490 - 440)
        B = 1.0
    elif wl < 510:
        R = 0.0
        G = 1.0
        B = -(wl - 510) / (510 - 490)
    elif wl < 580:
        R = (wl - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif wl < 645:
        R = 1.0
        G = -(wl - 645) / (645 - 580)
        B = 0.0
    else:
        R = 1.0
        G = 0.0
        B = 0.0

    if wl < 420:
        factor = 0.3 + 0.7 * (wl - 380) / (420 - 380)
    elif wl < 645:
        factor = 1.0
    else:
        factor = 0.3 + 0.7 * (780 - wl) / (780 - 645)

    R = round(R * factor * 255)
    G = round(G * factor * 255)
    B = round(B * factor * 255)

    return (R / 255, G / 255, B / 255)

def wavelength_band_label_nm(wl_nm):
        """Classify non-visible wavelengths into coarse EM-band labels.

        Parameters
        ----------
        wl_nm : float | None
            Wavelength in nanometers.

        Returns
        -------
        tuple[str | None, str | None]
            A tuple ``(label, color_hex)``. Visible wavelengths return ``(None, None)``
            because they are rendered with :func:`wavelength_to_rgb`.

        Notes
        -----
        The THz window (30 um to 3 mm) takes precedence over overlapping broader
        infrared/microwave naming ranges.
        """
        if wl_nm is None or not (wl_nm > 0):
            return None, None

        # Gamma
        if wl_nm < 1e-2:
            return "γ", "#666"

        # X-ray
        if wl_nm < 10.0:
            return "X", "#666"

        # EUV
        if wl_nm < 121.0:
            return "EUV", "#666"

        # UV
        if wl_nm < 380.0:
            return "UV", "#666"

        # Visible handled by the color swatch elsewhere
        if wl_nm <= 780.0:
            return None, None

        # THz priority window (30 μm–3 mm)
        if 3.0e4 <= wl_nm < 3.0e6:
            return "THz", "#666"

        # IR (0.78–30 μm), outside THz window
        if 780.0 <= wl_nm < 3.0e4:
            return "IR", "#666"

        # Microwave μ (1 mm–30 cm)
        if 1.0e6 <= wl_nm < 3.0e8:
            return "μ", "#666"

        # Radio RF (≥ 30 cm)
        if wl_nm >= 3.0e8:
            return "RF", "#666"

        # Fallback
        return None, None

def pretty_length(qty, digits=3):
    """Format a length quantity as a compact LaTeX fragment.

    Parameters
    ----------
    qty : astropy.units.Quantity
        Length quantity.
    digits : int, default=3
        Significant digits in the formatted value.

    Returns
    -------
    str
        A LaTeX snippet without surrounding ``$`` delimiters.

    Raises
    ------
    AttributeError
        If ``qty`` is not an astropy quantity-like object with ``to_value``.

    Notes
    -----
    Chooses the first unit in ``(m, cm, mm, um, nm)`` whose absolute numeric value
    lies in ``[1, 1000)``; otherwise falls back to meters.
    """
    for unit in (units.m, units.cm, units.mm, units.um, units.nm):
        v = qty.to_value(unit)
        if 1 <= abs(v) < 1000:
            unit_str = unit.to_string('latex').replace('$', '')  # remove $ signs
            return rf"{v:.{digits}g} \, {unit_str}"
    # fallback to metres
    unit_str = units.m.to_string('latex').replace('$', '')
    return rf"{qty.to_value(units.m):.{digits}g} \, {unit_str}"

def repr_html_modes(modes):
    """Build a single HTML table summarizing a sequence of guided modes.

    Parameters
    ----------
    modes : iterable
        Iterable of guided-mode-like objects with at least ``mode_label``, ``wl``,
        ``V``, ``neff``, ``a_plus`` and ``a_minus`` attributes. Optional attributes
        ``ell`` and ``mode_type`` are used to select polarization conventions for
        Stokes parameters.

    Returns
    -------
    str
        HTML markup for a styled table. Returns an empty string if no valid modes
        are provided.

    Notes
    -----
    For ``ell != 0``, normalized Stokes parameters are computed from ``a_plus`` and
    ``a_minus``. For ``ell == 0``, TE is rendered as linear ``S1=-1`` and all other
    ``ell == 0`` modes as ``S1=+1``.
    """
    import numpy as np

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
        try:
            wl_nm = m.wl.to_value('nm') if _HAS_UNITS else float(m.wl) * 1e9
        except Exception:
            wl_nm = float(m.wl) * 1e9

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
            S0 = np.abs(m.a_plus)**2 + np.abs(m.a_minus)**2
            S1 = 2 * np.real(m.a_plus * np.conj(m.a_minus))/S0
            S2 = 2 * np.imag(m.a_plus * np.conj(m.a_minus))/S0
            S3 = (np.abs(m.a_plus)**2 - np.abs(m.a_minus)**2)/S0
        elif getattr(m, "ell", 0) == 0 and getattr(m, "mode_type", None) == "TE":
            S0 = np.abs(m.a_plus)**2 + np.abs(m.a_minus)**2
            S1, S2, S3 = -1.0, 0.0, 0.0
        else:
            S0 = np.abs(m.a_plus)**2 + np.abs(m.a_minus)**2
            S1, S2, S3 = 1.0, 0.0, 0.0
        
        def fmt_signed(x):
            return f"{x:+.2f}".replace("+", "\u2008")

        rows.append(f"""
        <tr>
            <td style="text-align:center;">{m.mode_label}</td>
            <td class="num">{swatch_html}{wl_nm:.2f}</td>
            <td class="num">{m.V:.2f}</td>
            <td class="num">{m.neff:.4f}</td>
            <td class="num">{S0:.2f}</td>
            <td class="num">({fmt_signed(S1)}, {fmt_signed(S2)}, {fmt_signed(S3)})</td>
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
        <th scope="col" class="num"><i>λ</i> [nm]</th>
        <th scope="col" class="num"><i>V</i></th>
        <th scope="col" class="num"><i>n</i><sub>eff</sub></th>
        <th scope="col" class="num"><i>S</i><sub>0</sub></th>
        <th scope="col" class="num">(<i>S</i><sub>1</sub>, <i>S</i><sub>2</sub>, <i>S</i><sub>3</sub>) / <i>S</i><sub>0</sub></th>
        </tr>
    </thead>
    <tbody>
        {''.join(rows)}
    </tbody>
    </table>
    </div>
    """
    return html

class GuidedModeList(list):
    """List subclass that preserves rich HTML display for mode collections.

    Slicing returns another :class:`GuidedModeList` so notebook rendering via
    ``_repr_html_`` is retained for subsets.
    """

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if isinstance(item, slice):
            return GuidedModeList(result)
        return result

    def _repr_html_(self):
        return repr_html_modes(self)

def display_modes(*modes):
    """Render one or more guided modes as an HTML table in notebooks.

    Parameters
    ----------
    *modes
        One or more :class:`anafibre.fields.GuidedMode` objects.

    Returns
    -------
    None
        Displays HTML output via IPython.

    Notes
    -----
    Intended for notebook environments where ``IPython.display`` is available.
    """
    from IPython.display import display_html
    
    display_html(repr_html_modes(modes), raw=True)

def display_anim(anim):
    """Render a Matplotlib animation inline in notebooks.

    Parameters
    ----------
    anim : matplotlib.animation.Animation
        Animation instance with a ``to_jshtml`` method.

    Returns
    -------
    None
        Displays the animation via IPython HTML output.

    Notes
    -----
    Uses ``anim.to_jshtml()`` for inline playback in Jupyter-style frontends.
    """
    from IPython.display import HTML, display

    display(HTML(anim.to_jshtml()))
