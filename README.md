# hullform — parametric hull surfaces, hydrostatics and a Rhino model

A hull is defined once, as a half-breadth field `y(x, z)`, and everything else
is derived from it: the NURBS surface Rhino opens, the lines plan, the offset
table, the hydrostatics, the Bonjean curves, the righting-lever curve, and the
watertight mesh the CFD study in
[wigley-cfd](../wigley-cfd) runs on.

Nothing is read from a table of published coefficients. The point of the
project is that the numbers are *computed*, and then checked against a hull
whose answers are known exactly.

```bash
python -m hullform.cli build --hull wigley --L 100 --B 10 --T 6.25
python -m hullform.cli build --hull series --L 120 --B 18 --T 7 --Cp 0.62
python -m hullform.cli validate        # convergence against the exact values
pytest tests                            # 16 tests
```

## What it produces

| File | What it is |
|---|---|
| `wigley.3dm` | Rhino model: bicubic NURBS hull surface (both sides), stations, waterlines, buttocks and the DWL, on separate layers |
| `wigley.stl` | watertight triangulated half hull, for meshing in CFD |
| `wigley_offsets.csv` | the offset table, stations × waterlines |
| `wigley_hydrostatics.csv` | hydrostatic particulars over the draught range |
| `wigley_lines.png` | body plan, half-breadth plan, sheer plan |
| `wigley_curves_of_form.png` | displacement, coefficients, KB/KM, TPC |
| `wigley_bonjean.png` | Bonjean curves — sectional area against draught |
| `wigley_gz.png` | righting lever against heel, with the GM·sin φ tangent |

## Validation

The Wigley parabolic hull

```
y(x, z) = (B/2) · (1 − (2x/L)²) · (1 − (z/T)²)
```

has closed-form hydrostatics, so the quadratures can be checked rather than
merely inspected. With Simpson's rule on 201 ordinates in each direction:

| Quantity | Exact | Computed | Relative error |
|---|---|---|---|
| Block coefficient `Cb` | 4/9 = 0.444444444 | 0.444444444 | 1.2 × 10⁻¹⁶ |
| Waterplane coefficient `Cw` | 2/3 | 0.666666667 | 1.7 × 10⁻¹⁶ |
| Midship coefficient `Cm` | 2/3 | 0.666666667 | < 10⁻¹⁵ |
| Vertical centre of buoyancy `KB` | 5T/8 = 3.906250 m | 3.906250000 m | 0 |
| Metacentric radius `BMt` | 1.371428571 m | 1.371428563 m | 5.8 × 10⁻⁹ |
| Longitudinal centre `LCB` | 0 (amidships) | 1.7 × 10⁻¹⁵ m | — |

`Cb`, `Cw` and `KB` come back to machine precision because Simpson's rule is
exact for the polynomials involved. `BMt` involves `y³`, a sixth-order
polynomial in x, so it converges at fourth order instead — from 8.7 × 10⁻⁴ at
11 ordinates to 8.9 × 10⁻¹⁰ at 321, which is the line in `convergence.png`.

The stability side is checked the way a naval architect checks it: the initial
slope of the GZ curve must equal GM. Integrating the immersed volume of the
heeled hull gives, for KG = 0.65 T,

```
GM from KM − KG        1.2152 m
GZ(5°) / sin 5°        1.2072 m      (0.65 % apart)
GZ(2°) / sin 2°        1.2016 m
```

and the curve departs from the tangent above about 30°, as the form effects
take over.

## How the pieces work

**Geometry** (`geometry.py`) — a `Hull` is a half-breadth field plus principal
dimensions. `WigleyHull` is the analytic test form; `SeriesHull` is a
ship-like parametric form with a sectional-area curve, parallel middle body
and a section-shape exponent, so `Cp`, bilge fullness and the length of the
parallel middle body are inputs.

**Hydrostatics** (`hydrostatics.py`) — Simpson quadrature over the offsets for
volume, centres, waterplane inertia, TPC and MCT; Bonjean curves by
integrating each station separately; GZ by voxelising the hull once and
bisecting the heeled waterline until the immersed volume matches the upright
displacement.

**Rhino export** (`rhino_export.py`) — the surface is *interpolated*, not
approximated: global cubic B-spline interpolation (Piegl & Tiller A9.1, A9.4)
through the offset grid, so the surface passes through every offset. On a
100 m hull the fitted surface deviates from the analytic offsets by at most
**2.8 × 10⁻⁶ m**. The same module writes the watertight STL used by the CFD
study — welded by position, so the stem, stern and keel close properly and
`snappyHexMesh` can tell inside from outside.

## Requirements

`numpy`, `scipy`, `matplotlib`, `rhino3dm`, `pytest`. No Rhino licence is
needed to *write* the model; Rhino, or any viewer that reads `.3dm`, opens it.
