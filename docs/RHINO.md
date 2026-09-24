# Working with the model in Rhino

The generator writes a real Rhino file — `out/wigley.3dm` — not a mesh export.
Opening it gives a NURBS surface and the lines, each on its own layer, which
is what the rest of a hull-design workflow expects.

## What is in the file

| Layer | Contents |
|---|---|
| **Hull surface** | the fitted bicubic NURBS surface, port and starboard |
| **Stations** | transverse sections — the body plan |
| **Waterlines** | horizontal cuts — the half-breadth plan |
| **Buttocks** | longitudinal vertical cuts — the sheer plan |
| **DWL** | the design waterline, separated so it can be shown in red |

Units are metres, x is positive forward, z is positive up with zero at the
design waterline, and y is the half-breadth.

## Things worth doing once it is open

**Check the fairing.** `Zebra` or `EMap` on the hull surface: the stripes
should run smoothly with no kinks. A kink means the offsets that were fed in
were not fair, not that the fit is wrong — the surface interpolates them
exactly.

**Read the surface deviation.** `PointDeviation` with the station curves
against the surface reports how far the two disagree; it should be
microscopic, because both come from the same offsets.

**Make a drawing.** `Make2D` from Top, Front and Right gives the three views
of the lines plan as curves, ready to dimension and title.

**Check hydrostatics against another tool.** If Orca3D is installed, running
its hydrostatics on the surface should reproduce the numbers in
`wigley_hydrostatics.csv` — displacement 2 847 t, KB 3.906 m, KM 5.278 m for
the default 100 m Wigley hull. Disagreement means one of the two is wrong,
and the Wigley hull's exact values say which.

**Take it further.** `Loft`, `Sweep2` or a Grasshopper definition can rebuild
the same surface from the station curves, which is the usual way a hull is
faired by hand; the generated surface then serves as the target to compare
against.

## Why the surface is fitted, not lofted

Lofting through station curves gives a surface that follows the sections but
is free to wander between them. Global B-spline interpolation, which is what
`rhino_export.py` does, solves for the control net that makes the surface pass
through every offset in the grid — stations *and* waterlines. On the 100 m
Wigley hull the result sits within 2.8 µm of the analytic offsets.

## Licence note

Writing the file needs no Rhino licence: `rhino3dm`, McNeel's open-source
geometry library, does the writing. Rhino itself (or any `.3dm` viewer) is
only needed to open it.
