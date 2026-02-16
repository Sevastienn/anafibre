"""
utils.py

Utility functions and optional dependency management for Anafibre.
Includes unit handling and color conversion for wavelength visualisation.
"""

try:
    import astropy.units as units
    _HAS_UNITS = True
except ImportError:
    _HAS_UNITS = False
    units = None  # Gracefully handle absence of astropy

def _strip_unit(val, unit=None):
    """
    Converts an input value (float or astropy Quantity) to float in desired units.
    If val has no unit, returns as-is.
    If unit is given, converts to that unit.
    """
    if _HAS_UNITS and hasattr(val, 'unit'):
        if unit is not None:
            return val.to(unit).value
        return val.si.value
    return val

def wavelength_to_rgb(wavelength):
    """
    Convert a wavelength (in nm) to an approximate RGB color for visualisation.
    Wavelength range: 380–780 nm. Outside this range returns black.
    """
    if wavelength < 380 or wavelength > 780:
        return (0, 0, 0)  # Outside visible range

    if wavelength < 440:
        R = -(wavelength - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif wavelength < 490:
        R = 0.0
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif wavelength < 510:
        R = 0.0
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif wavelength < 580:
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif wavelength < 645:
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
        B = 0.0
    else:
        R = 1.0
        G = 0.0
        B = 0.0

    if wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif wavelength < 645:
        factor = 1.0
    else:
        factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 645)

    R = round(R * factor * 255)
    G = round(G * factor * 255)
    B = round(B * factor * 255)

    return (R / 255, G / 255, B / 255)

def wavelength_band_label_nm(wl_nm):
        """
        Map wavelength (in nm) → (short_label, bg_color_hex).
        Visible keeps using wavelength_to_rgb; everything else gets a labeled gray badge.

        Band edges used (common conventions, with priority where ranges overlap):
        γ-rays:        λ < 0.01 nm
        X-rays:        0.01–10 nm
        EUV:           10–121 nm
        UV:            121–380 nm
        Visible:       380–780 nm (handled elsewhere with color swatch)
        IR:            780–30,000 nm (0.78–30 μm)  [NIR+MIR]
        THz:           30,000–3,000,000 nm (30 μm–3 mm)  [overlaps FIR/sub-mm; we show THz]
        μ (Microwave): 1,000,000–300,000,000 nm (1 mm–0.3 m)
        RF (Radio):    ≥ 300,000,000 nm (≥ 0.3 m)

        Notes:
        - The THz window (30 μm–3 mm) intentionally overlaps FIR/sub-mm. We show "THz" there.
        - If you prefer FIR up to 1 mm, move the THz lower edge or adjust priorities below.
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
    """
    Return a pure LaTeX fragment for a length Quantity (assumed in metres).
    Ensures the unit string has no extra $.
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
    import numpy as np
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
        # try:
        #     wl_nm = m.wavelength.to_value('nm') if _HAS_UNITS else float(m.wavelength) * 1e9
        #     wl_str = f"{wl_nm:.2f}"
        # except Exception:
        #     wl_nm, wl_str = None, f"{m.wavelength:.2e}"
        try:
            wl_nm = m.wavelength.to_value('nm') if _HAS_UNITS else float(m.wavelength) * 1e9
        except Exception:
            wl_nm = float(m.wavelength) * 1e9

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

        rows.append(f"""
        <tr>
            <td style="text-align:center;">{m.mode_label()}</td>
            <td class="num">{swatch_html}{wl_nm:.2f}</td>
            <td class="num">{m.V:.2f}</td>
            <td class="num">{m.neff:.4f}</td>
            <td class="num">{S1:.2f}</td>
            <td class="num">{S2:.2f}</td>
            <td class="num">{S3:.2f}</td>
            <td class="num">{S0:.2f}</td>
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
        <th scope="col" class="num">S<sub>0</sub></th>
        </tr>
    </thead>
    <tbody>
        {''.join(rows)}
    </tbody>
    </table>
    </div>
    """
    return html