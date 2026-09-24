"""Rhino (.3dm) and STL export.

The hull is defined analytically, so the surface handed to Rhino is *fitted*,
not sampled: a bicubic B-spline surface is interpolated through the offset
grid with the standard global interpolation algorithm (Piegl & Tiller, The
NURBS Book, A9.1 and A9.4).  The result passes through every offset point,
which is what a fairing job needs -- an approximating surface would quietly
move the waterlines.

Layers written:

    Hull surface   the fitted bicubic surface, starboard and port
    Stations       transverse sections, the body plan
    Waterlines     horizontal cuts, the half-breadth plan
    Buttocks       longitudinal vertical cuts, the sheer plan
    DWL            the design waterline, marked separately
"""

from __future__ import annotations

import numpy as np
import rhino3dm as r

DEGREE = 3


# ---------------------------------------------------------------- B-splines


def _chord_params(points: np.ndarray) -> np.ndarray:
    """Centripetal parameterisation of a point row (NURBS Book eq. 9.6)."""
    d = np.sqrt(np.linalg.norm(np.diff(points, axis=0), axis=1))
    total = d.sum()
    if total <= 0:
        return np.linspace(0.0, 1.0, len(points))
    u = np.zeros(len(points))
    u[1:] = np.cumsum(d) / total
    u[-1] = 1.0
    return u


def _knots(u: np.ndarray, p: int = DEGREE) -> np.ndarray:
    """Averaging knot vector for interpolation (NURBS Book eq. 9.8)."""
    n = len(u) - 1
    m = n + p + 1
    kv = np.zeros(m + 1)
    kv[-(p + 1):] = 1.0
    for j in range(1, n - p + 1):
        kv[j + p] = u[j:j + p].mean()
    return kv


def _basis(i: int, p: int, u: float, kv: np.ndarray) -> float:
    """Cox-de Boor basis function N_{i,p}(u)."""
    if p == 0:
        return 1.0 if (kv[i] <= u < kv[i + 1]) or (u == kv[-1] and kv[i] < kv[i + 1] == kv[-1]) else 0.0
    left = 0.0
    if kv[i + p] > kv[i]:
        left = (u - kv[i]) / (kv[i + p] - kv[i]) * _basis(i, p - 1, u, kv)
    right = 0.0
    if kv[i + p + 1] > kv[i + 1]:
        right = (kv[i + p + 1] - u) / (kv[i + p + 1] - kv[i + 1]) * _basis(i + 1, p - 1, u, kv)
    return left + right


def _interp_matrix(u: np.ndarray, kv: np.ndarray, p: int = DEGREE) -> np.ndarray:
    n = len(u)
    A = np.zeros((n, n))
    for k, uk in enumerate(u):
        for i in range(n):
            A[k, i] = _basis(i, p, uk, kv)
    A[-1, -1] = 1.0  # close the clamped end exactly
    return A


def interpolate_curve(points: np.ndarray):
    """Control points and knots of the cubic B-spline through `points`."""
    u = _chord_params(points)
    kv = _knots(u)
    A = _interp_matrix(u, kv)
    cps = np.linalg.solve(A, points)
    return cps, kv


def interpolate_surface(grid: np.ndarray):
    """Bicubic interpolation of a (nu, nv, 3) point grid.

    Interpolates along v for every u-row, then along u through the resulting
    control points -- the standard two-pass construction.
    """
    nu, nv, _ = grid.shape
    # pass 1: along v
    tmp = np.zeros_like(grid)
    kv_v = None
    for i in range(nu):
        cps, kv_v = interpolate_curve(grid[i])
        tmp[i] = cps
    # pass 2: along u
    cp = np.zeros_like(grid)
    kv_u = None
    for j in range(nv):
        cps, kv_u = interpolate_curve(tmp[:, j, :])
        cp[:, j, :] = cps
    return cp, kv_u, kv_v


# ------------------------------------------------------------------ Rhino


def _nurbs_surface(cp: np.ndarray, kv_u: np.ndarray, kv_v: np.ndarray, mirror: bool = False):
    nu, nv, _ = cp.shape
    srf = r.NurbsSurface.Create(3, False, DEGREE + 1, DEGREE + 1, nu, nv)
    for i in range(nu):
        for j in range(nv):
            x, y, z = cp[i, j]
            srf.Points[i, j] = r.Point4d(float(x), float(-y if mirror else y), float(z), 1.0)
    for i, k in enumerate(kv_u[1:-1]):
        srf.KnotsU[i] = float(k)
    for j, k in enumerate(kv_v[1:-1]):
        srf.KnotsV[j] = float(k)
    return srf


def _curve(points: np.ndarray):
    pts = [r.Point3d(float(a), float(b), float(c)) for a, b, c in points]
    return r.NurbsCurve.Create(False, DEGREE, pts)


def _layer(model, name: str, colour):
    layer = r.Layer()
    layer.Name = name
    layer.Color = colour
    return model.Layers.Add(layer)


def write_3dm(
    hull,
    path: str,
    n_stations: int = 41,
    n_waterlines: int = 21,
    n_plan_stations: int = 21,
    n_plan_waterlines: int = 7,
    n_buttocks: int = 5,
):
    """Write the hull as a Rhino model: fitted surface plus the lines plan."""
    model = r.File3dm()
    model.ApplicationName = "hullform"
    model.ApplicationDetails = "parametric hull generator"

    li_surface = _layer(model, "Hull surface", (60, 90, 160, 255))
    li_stations = _layer(model, "Stations", (30, 30, 30, 255))
    li_waterlines = _layer(model, "Waterlines", (40, 120, 70, 255))
    li_buttocks = _layer(model, "Buttocks", (150, 80, 40, 255))
    li_dwl = _layer(model, "DWL", (200, 40, 40, 255))

    # -- fitted surface through the offsets
    x = hull.stations(n_stations)
    z = hull.waterlines(n_waterlines)
    y = hull.halfbreadth(x[:, None], z[None, :])
    grid = np.stack(
        [np.repeat(x[:, None], n_waterlines, axis=1), y, np.repeat(z[None, :], n_stations, axis=0)],
        axis=-1,
    )
    cp, kv_u, kv_v = interpolate_surface(grid)

    for mirror in (False, True):
        srf = _nurbs_surface(cp, kv_u, kv_v, mirror=mirror)
        att = r.ObjectAttributes()
        att.LayerIndex = li_surface
        att.Name = "hull_starboard" if not mirror else "hull_port"
        model.Objects.AddSurface(srf, att)

    # -- body plan: station sections
    zs = hull.waterlines(61)
    for xi in hull.stations(n_plan_stations):
        ys = hull.halfbreadth(np.full_like(zs, xi), zs)
        pts = np.stack([np.full_like(zs, xi), ys, zs], axis=-1)
        att = r.ObjectAttributes()
        att.LayerIndex = li_stations
        model.Objects.AddCurve(_curve(pts), att)

    # -- half-breadth plan: waterlines
    xs = hull.stations(121)
    for k, zi in enumerate(hull.waterlines(n_plan_waterlines)):
        ys = hull.halfbreadth(xs, np.full_like(xs, zi))
        pts = np.stack([xs, ys, np.full_like(xs, zi)], axis=-1)
        att = r.ObjectAttributes()
        att.LayerIndex = li_dwl if abs(zi) < 1e-12 else li_waterlines
        model.Objects.AddCurve(_curve(pts), att)

    # -- sheer plan: buttocks, cut at constant y
    for yb in np.linspace(hull.B / 2 / (n_buttocks + 1), hull.B / 2 * 0.9, n_buttocks):
        pts = []
        for xi in xs:
            zz = hull.waterlines(121)
            yy = hull.halfbreadth(np.full_like(zz, xi), zz)
            if yy.max() < yb:
                continue
            zi = np.interp(yb, yy, zz)  # yy rises monotonically with z
            pts.append((xi, yb, zi))
        if len(pts) > DEGREE:
            att = r.ObjectAttributes()
            att.LayerIndex = li_buttocks
            model.Objects.AddCurve(_curve(np.array(pts)), att)

    model.Write(path, 7)
    return path


# -------------------------------------------------------------------- STL


def hull_mesh(hull, n_stations: int = 161, n_waterlines: int = 81, z_top: float | None = None):
    """A closed triangular mesh of the half hull, for CFD.

    The hull is extruded vertically from the design waterline to z_top (the
    Wigley form is only defined to the DWL), then closed with a centreplane
    at y = 0 and a deck at z_top, so the result is watertight -- which is what
    snappyHexMesh needs to tell inside from outside.
    """
    z_top = 0.5 * hull.T if z_top is None else z_top
    x = hull.stations(n_stations)
    z_hull = hull.waterlines(n_waterlines)
    z_ext = np.linspace(0.0, z_top, max(3, n_waterlines // 4))
    z = np.concatenate([z_hull, z_ext[1:]])

    y = np.zeros((len(x), len(z)))
    y[:, : len(z_hull)] = hull.halfbreadth(x[:, None], z_hull[None, :])
    y[:, len(z_hull):] = y[:, len(z_hull) - 1][:, None]  # vertical wall above DWL

    nx, nz = len(x), len(z)

    # Vertices are welded by position, so the shell, the centreplane and the
    # deck share the points they touch.  Without this the stem and stern --
    # where the offsets collapse to y = 0 -- leave a seam of open edges, and
    # snappyHexMesh cannot tell inside from outside at a leaky surface.
    verts: list[tuple[float, float, float]] = []
    index: dict[tuple[int, int, int], int] = {}
    scale = 1e6

    def vid(px, py, pz):
        key = (int(round(px * scale)), int(round(py * scale)), int(round(pz * scale)))
        if key not in index:
            index[key] = len(verts)
            verts.append((float(px), float(py), float(pz)))
        return index[key]

    faces = []

    def quad(p0, p1, p2, p3):
        a, b, c, d = (vid(*p) for p in (p0, p1, p2, p3))
        if len({a, b, c}) == 3:
            faces.append((a, b, c))
        if len({a, c, d}) == 3:
            faces.append((a, c, d))

    # the shell
    for i in range(nx - 1):
        for j in range(nz - 1):
            quad(
                (x[i], y[i, j], z[j]),
                (x[i + 1], y[i + 1, j], z[j]),
                (x[i + 1], y[i + 1, j + 1], z[j + 1]),
                (x[i], y[i, j + 1], z[j + 1]),
            )

    # the centreplane, on the same (x, z) grid so every edge has a partner
    for i in range(nx - 1):
        for j in range(nz - 1):
            quad(
                (x[i], 0.0, z[j]),
                (x[i], 0.0, z[j + 1]),
                (x[i + 1], 0.0, z[j + 1]),
                (x[i + 1], 0.0, z[j]),
            )

    # the deck, closing the top between the shell edge and the centreline
    for i in range(nx - 1):
        quad(
            (x[i], y[i, nz - 1], z[-1]),
            (x[i + 1], y[i + 1, nz - 1], z[-1]),
            (x[i + 1], 0.0, z[-1]),
            (x[i], 0.0, z[-1]),
        )

    return np.array(verts), np.array(faces)


def write_stl(hull, path: str, name: str = "hull", **kwargs):
    """Write the half hull as a binary-free ASCII STL (readable, diffable)."""
    verts, faces = hull_mesh(hull, **kwargs)
    with open(path, "w") as fh:
        fh.write(f"solid {name}\n")
        for f in faces:
            p0, p1, p2 = verts[f[0]], verts[f[1]], verts[f[2]]
            n = np.cross(p1 - p0, p2 - p0)
            ln = np.linalg.norm(n)
            n = n / ln if ln > 0 else np.array([0.0, 0.0, 1.0])
            fh.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n    outer loop\n")
            for p in (p0, p1, p2):
                fh.write(f"      vertex {p[0]:.6e} {p[1]:.6e} {p[2]:.6e}\n")
            fh.write("    endloop\n  endfacet\n")
        fh.write(f"endsolid {name}\n")
    return path
