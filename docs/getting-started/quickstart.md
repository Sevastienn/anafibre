# Quickstart

This notebook provides a quick start guide to using Anafibre. It covers the basics of:

- creating a fibre, 
- calculating its modes, and 
- visualising the results.

## Imports

We will start by importing the necessary libraries and setting up our environment:

```python
import numpy as np
import anafibre as fib
```

## Create A Step-Index Fibre
Next, we will create a step-index fibre class instance with a core radius of $250 \text{ nm}$, a core refractive index of $2.00$, and a cladding refractive index of $1.33$:
```python
fibre = fib.StepIndexFibre(
    core_radius=250e-9,
    n_core=2.00,
    n_clad=1.33)
```
and choose a wavelength of $\lambda=700 \text{ nm}$ for the mode calculations:
```python
wl = 700e-9
```
Notice that all the parameters are specified in SI units such as the core radius and wavelength in meters. Refractive indices `n_core` and `n_clad` can be specified as functions of wavelength, or as constants like in the example above. If we wanted a function we could pass `n_core=lambda x: 2.00+0.01*(x-700e-9)` for example, which would give a core refractive index that increases linearly with wavelength. One can also instead specify the relative `eps_core` and `eps_clad` parameters, and `mu_core` and `mu_clad` if the fibre is magnetic. The fibre object will automatically calculate the missing parameters from the ones that are provided. One can call fibre parameters at any wavelength as shown below:
```python
print(f"Fibre parameters at lambda = {wl*1e9:.0f} nm:")
print(f"  Core radius: {fibre.core_radius*1e9:.0f} nm")
print(f"  n_core:      {fibre.n_core(wl):.2f}")
print(f"  n_clad:      {fibre.n_clad(wl):.2f}")
print(f"\nMaterial properties:")
print(f"  eps_core:    {fibre.eps_core(wl):.2f}")
print(f"  eps_clad:    {fibre.eps_clad(wl):.2f}")
print(f"  mu_core:     {fibre.mu_core(wl):.2f}")
print(f"  mu_clad:     {fibre.mu_clad(wl):.2f}")
print(f"\nV-number at lambda = {wl*1e9:.0f} nm: {fibre.V(wl):.3f}")
```

Output:

```text
Fibre parameters at lambda = 700 nm:
  Core radius: 250 nm
  n_core:      2.00
  n_clad:      1.33

Material properties:
  eps_core:    4.00
  eps_clad:    1.77
  mu_core:     1.00
  mu_clad:     1.00

V-number at lambda = 700 nm: 3.352
```

## Calculate guided modes
The fibre class also  contains **mode constructor** methods which can be used to calculate the guided modes supported by the fibre at a given wavelength. For step-index fibres, these include methods for calculating the
 $\text{HE}_{\ell n}$, $\text{EH}_{\ell n}$, $\text{TE}_{0n}$, and $\text{TM}_{0n}$ modes
>```python
>fibre.HE(ell, n, wl, a_plus=..., a_minus=...)
>fibre.EH(...)
>fibre.TE(n, wl, a=...)
>fibre.TM(...)
>```

Each returns a `GuidedMode` object. So for instance, to calculate the fundamental $\text{HE}_{11}$ mode at our chosen wavelength and with linear polarisation along the x-axis, we can do:

```python
HE11x = fibre.HE(ell=1, n=1, wl=wl, a_plus=1/np.sqrt(2), a_minus=1/np.sqrt(2))
HE11x
```

Output:

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
        .anafibre-guidedmode table {
            width: auto;
            background: var(--bg);
            color: var(--fg);
            border: 1px solid var(--border);
            border-radius: 10px;
            border-collapse: separate;
            border-spacing: 0;
            overflow: hidden;
            box-shadow: 0 2px 10px color-mix(in srgb, var(--fg) 10%, transparent);
        }
        .anafibre-guidedmode th,
        .anafibre-guidedmode td {
            padding: .55rem .5rem;
            vertical-align: middle;
        }
        .anafibre-guidedmode thead th {
            background: var(--header);
            font-weight: 600;
        }
        .anafibre-guidedmode td.num,
        .anafibre-guidedmode th.num {
            text-align: center;
        }
        .anafibre-guidedmode tbody tr:nth-child(odd) td {
            background: var(--row);
        }
    </style>

    <table>
        <thead>
            <tr>
                <th scope="col" style="text-align:center;">Mode</th>
                <th scope="col" class="num"><i>λ</i> [nm]</th>
                <th scope="col" class="num"><i>V</i></th>
                <th scope="col" class="num"><i>n</i><sub>eff</sub></th>
                <th scope="col" class="num"><i>S</i><sub>0</sub></th>
                <th scope="col" class="num">
                    (<i>S</i><sub>1</sub>, <i>S</i><sub>2</sub>, <i>S</i><sub>3</sub>) / <i>S</i><sub>0</sub>
                </th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="text-align:center;">HE<sub>11</sub></td>
                <td class="num">
                    <span style="
                        display:inline-flex;
                        align-items:center;
                        justify-content:center;
                        width:1.5em;
                        height:1.5em;
                        margin-right:.4em;
                        margin-bottom:0.3em;
                        border:1px solid rgba(0,0,0,.25);
                        border-radius:2px;
                        background:rgb(182, 0, 0);
                        color:white;
                        font-size:clamp(0.45em, 0.55em, 0.65em);
                        font-weight:700;
                        letter-spacing:.02em;
                        vertical-align:middle;
                        overflow:hidden;
                        white-space:nowrap;
                        text-overflow:ellipsis;
                    "></span>
                    700.00
                </td>
                <td class="num">3.35</td>
                <td class="num">1.7955</td>
                <td class="num">1.00</td>
                <td class="num">( 1.00,  0.00,  0.00)</td>
            </tr>
        </tbody>
    </table>
</div>

The mode class provides methods to calculate the fields at any point in space, for example the electric field at the fibre centre:

```python
E_center = HE11x.E(x=0, y=0)
print("Electric field at the fibre center (0,0,0):")
print(f" Ex = {E_center[0].real:+.2e} {E_center[0].imag:+.2e} i")
print(f" Ey = {E_center[1].real:+.2e} {E_center[1].imag:+.2e} i")
print(f" Ez = {E_center[2].real:+.2e} {E_center[2].imag:+.2e} i")
print(f"\n |E| = {np.abs(E_center).sum():.2e} V/m")
```

Output:

```text
Electric field at the fibre center (0,0,0):
 Ex = +0.00e+00 +6.68e+07 i
 Ey = +0.00e+00 +0.00e+00 i
 Ez = +0.00e+00 +0.00e+00 i

|E| = 6.68e+07 V/m
```

We can also list all the guided modes at this wavelength using the mode listing method:

```python
modes = fibre.list_modes_at(wl)
modes
```

Output:

<div class="anafibre-guidedmode" aria-label="Guided mode summary"
        style="color-scheme: light dark; --bg: Canvas; --fg: CanvasText;
                --border: color-mix(in srgb, var(--fg) 18%, var(--bg));
                --header: color-mix(in srgb, var(--fg) 8%, var(--bg));
                --row: color-mix(in srgb, var(--fg) 4%, var(--bg));
                font-variant-numeric: tabular-nums;">
    <style>
    .anafibre-guidedmode table {
        width: auto; background: var(--bg); color: var(--fg);
        border: 1px solid var(--border); border-radius: 10px;
        border-collapse: separate; border-spacing: 0; overflow: hidden;
        box-shadow: 0 2px 10px color-mix(in srgb, var(--fg) 10%, transparent);
    }
    .anafibre-guidedmode th, .anafibre-guidedmode td {
        padding: .55rem .5rem; vertical-align: middle;
    }
    .anafibre-guidedmode thead th {
        background: var(--header); font-weight: 600;
    }
    .anafibre-guidedmode td.num, .anafibre-guidedmode th.num { text-align: center; }
    .anafibre-guidedmode tbody tr:nth-child(odd) td { background: var(--row); }
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
            <td style="text-align:center;">HE<sub>11</sub></td>
            <td class="num"><span style="display:inline-flex;align-items:center;justify-content:center;width:1.5em;height:1.5em;margin-right:.4em;margin-bottom:0.3em;border:1px solid rgba(0,0,0,.25);border-radius:2px;background:rgb(182, 0, 0);color:white;font-size:clamp(0.45em, 0.55em, 0.65em);font-weight:700;letter-spacing:.02em;vertical-align:middle;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;"></span>700.00</td>
            <td class="num">3.35</td>
            <td class="num">1.7955</td>
            <td class="num">1.00</td>
            <td class="num">( 0.00,  0.00,  1.00)</td>
        </tr>
        
        <tr>
            <td style="text-align:center;">TM<sub>01</sub></td>
            <td class="num"><span style="display:inline-flex;align-items:center;justify-content:center;width:1.5em;height:1.5em;margin-right:.4em;margin-bottom:0.3em;border:1px solid rgba(0,0,0,.25);border-radius:2px;background:rgb(182, 0, 0);color:white;font-size:clamp(0.45em, 0.55em, 0.65em);font-weight:700;letter-spacing:.02em;vertical-align:middle;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;"></span>700.00</td>
            <td class="num">3.35</td>
            <td class="num">1.5500</td>
            <td class="num">1.00</td>
            <td class="num">( 1.00,  0.00,  0.00)</td>
        </tr>
        
        <tr>
            <td style="text-align:center;">TE<sub>01</sub></td>
            <td class="num"><span style="display:inline-flex;align-items:center;justify-content:center;width:1.5em;height:1.5em;margin-right:.4em;margin-bottom:0.3em;border:1px solid rgba(0,0,0,.25);border-radius:2px;background:rgb(182, 0, 0);color:white;font-size:clamp(0.45em, 0.55em, 0.65em);font-weight:700;letter-spacing:.02em;vertical-align:middle;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;"></span>700.00</td>
            <td class="num">3.35</td>
            <td class="num">1.4762</td>
            <td class="num">1.00</td>
            <td class="num">(-1.00,  0.00,  0.00)</td>
        </tr>
        
        <tr>
            <td style="text-align:center;">HE<sub>21</sub></td>
            <td class="num"><span style="display:inline-flex;align-items:center;justify-content:center;width:1.5em;height:1.5em;margin-right:.4em;margin-bottom:0.3em;border:1px solid rgba(0,0,0,.25);border-radius:2px;background:rgb(182, 0, 0);color:white;font-size:clamp(0.45em, 0.55em, 0.65em);font-weight:700;letter-spacing:.02em;vertical-align:middle;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;"></span>700.00</td>
            <td class="num">3.35</td>
            <td class="num">1.4571</td>
            <td class="num">1.00</td>
            <td class="num">( 0.00,  0.00,  1.00)</td>
        </tr>
        
    </tbody>
    </table>
    </div>

Where hybrid modes will default to right-handed circular polarisation. It returns a list `modes` of `GuidedMode` objects. 


## Evaluate Fields On A Grid
Since $\vc{E}(\vc{r})$, $\vc{H}(\vc{r})$, $\grad\otimes\vc{E}(\vc{r})$, and $\grad\otimes\vc{H}(\vc{r})$ methods can be calculated at any point in space we can also evaluate them on a grid:

```python
mode = modes[0]
Ngrid = 100
x = np.linspace(-2*fibre.core_radius, 2*fibre.core_radius, Ngrid)
y = np.linspace(-2*fibre.core_radius, 2*fibre.core_radius, Ngrid)
X, Y = np.meshgrid(x, y)

E = mode.E(x=X, y=Y)
H = mode.H(x=X, y=Y)
J_E = mode.gradE(x=X, y=Y)
J_H = mode.gradH(x=X, y=Y)

print(f"Field shapes:\nE:{E.shape}, H:{H.shape}, gradE:{J_E.shape}, gradH:{J_H.shape}")
```

Output:

```text
Field shapes:
E:(100, 100, 3), H:(100, 100, 3), gradE:(100, 100, 3, 3), gradH:(100, 100, 3, 3)
```

## Visualise Modes

```python
anim = fib.animate_fields_xy(modes=modes[0], show=("E",), figsize=(5, 5))
fib.display_anim(anim)
```

The animation is interactive in Jupyter. In docs, link to the notebook for full rendered widget output:
- `examples/quick_start.ipynb`

To save animation files:

```python
anim.save("mode_animation.mp4", writer="ffmpeg", fps=30)
anim.save("mode_animation.gif", writer="pillow", fps=15)
```
