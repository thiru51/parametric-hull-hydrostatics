#!/usr/bin/env python
"""Render the fitted NURBS surface from a .3dm, as a picture for the README.

    python tools/render_surface.py out/wigley.3dm out/wigley_surface.png

Reads the surface back out of the Rhino file rather than re-evaluating the
hull formula, so the picture shows what was actually written.
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rhino3dm as r


def render(path_3dm: str, path_png: str, nu: int = 80, nv: int = 30):
    model = r.File3dm.Read(path_3dm)
    surfaces = [o.Geometry for o in model.Objects if o.Geometry.ObjectType == r.ObjectType.Surface]
    if not surfaces:
        raise SystemExit(f"no surfaces in {path_3dm}")

    fig = plt.figure(figsize=(11, 4.2))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    for s in surfaces:
        du, dv = s.Domain(0), s.Domain(1)
        u = np.linspace(du.T0, du.T1, nu)
        v = np.linspace(dv.T0, dv.T1, nv)
        P = np.empty((nu, nv, 3))
        for i, ui in enumerate(u):
            for j, vj in enumerate(v):
                p = s.PointAt(ui, vj)
                P[i, j] = (p.X, p.Y, p.Z)
        ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], rstride=1, cstride=1,
                        color="#5b86c9", edgecolor="#22436f", linewidth=0.12, alpha=0.95)

    ax.set_box_aspect((6, 1.1, 0.9), zoom=1.35)
    ax.set_xlabel("x, m", labelpad=-4)
    ax.set_ylabel("y, m", labelpad=-6)
    ax.set_zlabel("z, m", labelpad=-6)
    ax.tick_params(labelsize=7, pad=-2)
    ax.view_init(elev=24, azim=-142)
    ax.set_title("Fitted bicubic NURBS hull surface, read back from the Rhino model", pad=0)
    fig.tight_layout()
    fig.savefig(path_png, dpi=160, bbox_inches="tight")
    print("wrote", path_png)


if __name__ == "__main__":
    a = sys.argv[1:] or ["out/wigley.3dm", "out/wigley_surface.png"]
    render(a[0], a[1])
