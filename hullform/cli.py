"""Command line: build a hull, and write everything that comes with it.

    python -m hullform.cli build --hull wigley --L 100 --B 10 --T 6.25
    python -m hullform.cli validate
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from . import plots, rhino_export
from .geometry import SeriesHull, WigleyHull, build
from .hydrostatics import HydrostaticSolver


def _write_offsets(hull, path: str, n_stations: int = 21, n_waterlines: int = 11):
    x, z, y = hull.offsets(n_stations, n_waterlines)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["station_x_m"] + [f"z={zi:.3f}" for zi in z])
        for i, xi in enumerate(x):
            w.writerow([f"{xi:.4f}"] + [f"{v:.5f}" for v in y[i]])
    return path


def _write_hydrostatics(rows, draughts, path: str):
    fields = list(rows[0].as_dict().keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: f"{v:.6f}" for k, v in r.as_dict().items()})
    return path


def cmd_build(args):
    out = args.out
    os.makedirs(out, exist_ok=True)
    kwargs = dict(L=args.L, B=args.B, T=args.T)
    if args.hull == "series":
        kwargs.update(n_sec=args.n_sec, pmb=args.pmb, Cp_aim=args.Cp)
    hull = build(args.hull, **kwargs)

    solver = HydrostaticSolver(hull, n_stations=args.ordinates, n_waterlines=args.ordinates)
    design = solver.at_draught()
    draughts, rows = solver.curves_of_form(args.n_curves)
    xb, tb, Ab = solver.bonjean()

    KG = args.KG if args.KG is not None else 0.65 * hull.T
    heels, gz = solver.gz_curve(KG=KG, heels_deg=np.arange(0.0, 61.0, 5.0))
    GM = design.KMt - KG

    rhino_export.write_3dm(hull, os.path.join(out, f"{hull.name}.3dm"))
    rhino_export.write_stl(hull, os.path.join(out, f"{hull.name}.stl"))
    _write_offsets(hull, os.path.join(out, f"{hull.name}_offsets.csv"))
    _write_hydrostatics(rows, draughts, os.path.join(out, f"{hull.name}_hydrostatics.csv"))

    plots.lines_plan(hull, os.path.join(out, f"{hull.name}_lines.png"))
    plots.curves_of_form(draughts, rows, os.path.join(out, f"{hull.name}_curves_of_form.png"))
    plots.bonjean(xb, tb, Ab, os.path.join(out, f"{hull.name}_bonjean.png"))
    plots.gz_curve(heels, gz, KG, os.path.join(out, f"{hull.name}_gz.png"), GM=GM)

    summary = {
        "hull": hull.name,
        "L": hull.L,
        "B": hull.B,
        "T": hull.T,
        "design": design.as_dict(),
        "KG": KG,
        "GM": GM,
        "GZ_max_m": float(np.max(gz)),
        "GZ_max_at_deg": float(heels[int(np.argmax(gz))]),
        "area_to_30deg_m_rad": float(np.trapezoid(gz[heels <= 30], np.radians(heels[heels <= 30]))),
    }
    with open(os.path.join(out, f"{hull.name}_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


def cmd_validate(args):
    """Convergence of the quadrature against the Wigley hull's exact values."""
    hull = WigleyHull()
    ns = [11, 21, 41, 81, 161, 321]
    err = {"Cb": [], "Cw": [], "KB": [], "BMt": []}
    for n in ns:
        s = HydrostaticSolver(hull, n_stations=n + 1 if n % 2 == 0 else n,
                              n_waterlines=n + 1 if n % 2 == 0 else n)
        r = s.at_draught()
        err["Cb"].append(abs(r.Cb - 4 / 9) / (4 / 9))
        err["Cw"].append(abs(r.Cw - 2 / 3) / (2 / 3))
        err["KB"].append(abs(r.KB - hull.exact_KB()) / hull.exact_KB())
        err["BMt"].append(abs(r.BMt - hull.exact_BMt()) / hull.exact_BMt())
    os.makedirs(args.out, exist_ok=True)
    plots.convergence(ns, err, os.path.join(args.out, "convergence.png"))
    print(f"{'N':>5}  {'Cb':>12} {'Cw':>12} {'KB':>12} {'BMt':>12}")
    for i, n in enumerate(ns):
        print(f"{n:>5}  " + " ".join(f"{err[k][i]:12.3e}" for k in ("Cb", "Cw", "KB", "BMt")))
    return err


def main(argv=None):
    p = argparse.ArgumentParser(prog="hullform")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build a hull and write the model, drawings and hydrostatics")
    b.add_argument("--hull", default="wigley", choices=["wigley", "series"])
    b.add_argument("--L", type=float, default=100.0)
    b.add_argument("--B", type=float, default=10.0)
    b.add_argument("--T", type=float, default=6.25)
    b.add_argument("--n-sec", dest="n_sec", type=float, default=2.2)
    b.add_argument("--pmb", type=float, default=0.18)
    b.add_argument("--Cp", type=float, default=0.62)
    b.add_argument("--KG", type=float, default=None)
    b.add_argument("--ordinates", type=int, default=201)
    b.add_argument("--n-curves", dest="n_curves", type=int, default=21)
    b.add_argument("--out", default="out")
    b.set_defaults(func=cmd_build)

    v = sub.add_parser("validate", help="convergence against the Wigley hull's exact coefficients")
    v.add_argument("--out", default="out")
    v.set_defaults(func=cmd_validate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
