"""Drawings: the lines plan, the curves of form, Bonjean curves, the GZ curve."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLE = {"lw": 0.9}


def lines_plan(hull, path: str, n_stations: int = 21, n_waterlines: int = 7, n_buttocks: int = 5):
    """The three views a hull is faired in: body, half-breadth, sheer."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    fig.suptitle(
        f"Lines plan — {hull.name}   L {hull.L:.1f} m · B {hull.B:.1f} m · T {hull.T:.2f} m",
        fontsize=12,
    )

    # body plan: sections, aft half mirrored to the left as is conventional
    ax = axes[0]
    z = hull.waterlines(121)
    for xi in hull.stations(n_stations):
        y = hull.halfbreadth(np.full_like(z, xi), z)
        sign = 1.0 if xi >= 0 else -1.0
        ax.plot(sign * y, z, color="0.2", **STYLE)
    ax.axvline(0, color="0.6", lw=0.6)
    ax.axhline(0, color="crimson", lw=0.8)
    ax.set_title("Body plan (forward sections right, aft left)", fontsize=9)
    ax.set_xlabel("half-breadth, m")
    ax.set_ylabel("z, m")
    ax.set_aspect("equal")

    # half-breadth plan: waterlines
    ax = axes[1]
    x = hull.stations(241)
    for zi in hull.waterlines(n_waterlines):
        y = hull.halfbreadth(x, np.full_like(x, zi))
        ax.plot(x, y, color="seagreen" if abs(zi) > 1e-9 else "crimson", **STYLE)
    ax.set_title("Half-breadth plan (design waterline in red)", fontsize=9)
    ax.set_xlabel("x from midships, m")
    ax.set_ylabel("y, m")
    ax.set_aspect("equal")

    # sheer plan: buttocks
    ax = axes[2]
    zz = hull.waterlines(241)
    for yb in np.linspace(hull.B / 2 / (n_buttocks + 1), 0.9 * hull.B / 2, n_buttocks):
        pts = []
        for xi in x:
            y = hull.halfbreadth(np.full_like(zz, xi), zz)
            if y.max() >= yb:
                pts.append((xi, np.interp(yb, y, zz)))
        if pts:
            pts = np.array(pts)
            ax.plot(pts[:, 0], pts[:, 1], color="chocolate", **STYLE)
    ax.axhline(0, color="crimson", lw=0.8)
    ax.set_title("Sheer plan (buttock lines)", fontsize=9)
    ax.set_xlabel("x from midships, m")
    ax.set_ylabel("z, m")
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def curves_of_form(draughts, rows, path: str):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    disp = [r.displacement for r in rows]
    axes[0].plot(disp, draughts, color="navy")
    axes[0].set_xlabel("displacement, t")
    axes[0].set_ylabel("draught, m")
    axes[0].set_title("Displacement")

    axes[1].plot([r.Cb for r in rows], draughts, label="Cb")
    axes[1].plot([r.Cw for r in rows], draughts, label="Cw")
    axes[1].plot([r.Cp for r in rows], draughts, label="Cp")
    axes[1].legend(fontsize=7)
    axes[1].set_xlabel("coefficient")
    axes[1].set_title("Form coefficients")

    axes[2].plot([r.KB for r in rows], draughts, label="KB")
    axes[2].plot([r.KMt for r in rows], draughts, label="KMt")
    axes[2].legend(fontsize=7)
    axes[2].set_xlabel("m above keel")
    axes[2].set_title("KB and KM")

    axes[3].plot([r.TPC for r in rows], draughts, color="seagreen")
    axes[3].set_xlabel("TPC, t/cm")
    axes[3].set_title("Tonnes per cm")

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Curves of form", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def bonjean(x, draughts, A, path: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    for k, xi in enumerate(x):
        # each station's area curve, drawn at its own station line
        scale = (x[1] - x[0]) * 0.9 / max(A.max(), 1e-9)
        ax.plot(xi + A[k] * scale, draughts, color="0.25", lw=0.8)
        ax.axvline(xi, color="0.85", lw=0.5)
    ax.set_xlabel("station position, m  (area plotted to the right of each station)")
    ax.set_ylabel("draught, m")
    ax.set_title("Bonjean curves — immersed sectional area against draught")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def gz_curve(heels, gz, KG: float, path: str, GM: float | None = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(heels, gz, marker="o", ms=3, color="navy", label="GZ from immersed-volume integration")
    if GM is not None:
        ax.plot(heels, GM * np.sin(np.radians(heels)), "--", color="crimson", lw=1,
                label=f"GM·sin φ  (GM = {GM:.3f} m)")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel("heel angle φ, degrees")
    ax.set_ylabel("righting lever GZ, m")
    ax.set_title(f"Righting lever curve, KG = {KG:.2f} m")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def convergence(ns, errors, path: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, e in errors.items():
        ax.loglog(ns, e, marker="o", ms=3, label=label)
    ax.set_xlabel("ordinates per direction")
    ax.set_ylabel("relative error against the analytic value")
    ax.set_title("Convergence of the hydrostatics quadrature (Wigley hull)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
