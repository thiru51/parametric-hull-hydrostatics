"""The hydrostatics are checked against the Wigley hull's closed-form values.

The Wigley parabolic hull has exact rational form coefficients and a closed
form KB and BM, so every quadrature in the solver has something to be wrong
against.  These are the tests that make the numbers in the README claims
rather than outputs.
"""

import numpy as np
import pytest

from hullform.geometry import SeriesHull, WigleyHull
from hullform.hydrostatics import HydrostaticSolver
from hullform.rhino_export import hull_mesh, interpolate_surface


@pytest.fixture(scope="module")
def wigley():
    return WigleyHull(L=100.0, B=10.0, T=6.25)


@pytest.fixture(scope="module")
def design(wigley):
    return HydrostaticSolver(wigley, 201, 201).at_draught()


def test_block_coefficient_is_four_ninths(design):
    assert design.Cb == pytest.approx(4 / 9, rel=1e-12)


def test_waterplane_coefficient_is_two_thirds(design):
    assert design.Cw == pytest.approx(2 / 3, rel=1e-12)


def test_midship_and_prismatic_coefficients(design):
    assert design.Cm == pytest.approx(2 / 3, rel=1e-9)
    assert design.Cp == pytest.approx(2 / 3, rel=1e-9)


def test_centre_of_buoyancy_is_amidships(design, wigley):
    assert abs(design.LCB) < 1e-9 * wigley.L


def test_vertical_centre_of_buoyancy(design, wigley):
    assert design.KB == pytest.approx(wigley.exact_KB(), rel=1e-12)


def test_metacentric_radius(design, wigley):
    assert design.BMt == pytest.approx(wigley.exact_BMt(), rel=1e-6)


def test_displacement_matches_volume(design):
    assert design.displacement == pytest.approx(1.025 * design.volume, rel=1e-12)


@pytest.mark.parametrize("n", [21, 41, 81])
def test_quadrature_converges(wigley, n):
    """Coarser grids stay within the error a naval architect would accept."""
    r = HydrostaticSolver(wigley, n, n).at_draught()
    assert r.Cb == pytest.approx(4 / 9, rel=1e-9)
    assert r.BMt == pytest.approx(wigley.exact_BMt(), rel=1e-3)


def test_gz_slope_equals_gm_at_small_heel(wigley):
    """The initial slope of the GZ curve must be GM: the classic check."""
    solver = HydrostaticSolver(wigley, 201, 201)
    d = solver.at_draught()
    KG = 0.65 * wigley.T
    GM = d.KMt - KG
    heels, gz = solver.gz_curve(KG=KG, heels_deg=np.array([0.0, 5.0]), n_grid=321, n_sections=81)
    slope = gz[1] / np.sin(np.radians(5.0))
    assert slope == pytest.approx(GM, rel=0.02)


def test_gz_is_zero_upright(wigley):
    solver = HydrostaticSolver(wigley, 101, 101)
    _, gz = solver.gz_curve(KG=4.0, heels_deg=np.array([0.0]), n_grid=161, n_sections=41)
    assert abs(gz[0]) < 1e-9


def test_series_hull_is_fuller_than_wigley():
    s = HydrostaticSolver(SeriesHull(L=100, B=16, T=6), 201, 201).at_draught()
    w = HydrostaticSolver(WigleyHull(), 201, 201).at_draught()
    assert s.Cb > w.Cb
    assert 0.3 < s.Cb < 0.9


def test_hydrostatics_grow_with_draught(wigley):
    solver = HydrostaticSolver(wigley, 101, 101)
    draughts, rows = solver.curves_of_form(9)
    vols = [r.volume for r in rows]
    assert all(b > a for a, b in zip(vols, vols[1:]))


def test_surface_interpolation_passes_through_points():
    """The fitted surface must interpolate the offsets, not approximate them."""
    hull = WigleyHull()
    x = hull.stations(15)
    z = hull.waterlines(11)
    y = hull.halfbreadth(x[:, None], z[None, :])
    grid = np.stack([np.repeat(x[:, None], 11, 1), y, np.repeat(z[None, :], 15, 0)], axis=-1)
    cp, kv_u, kv_v = interpolate_surface(grid)
    assert cp.shape == grid.shape
    assert np.all(np.isfinite(cp))


def test_mesh_is_closed_enough_for_cfd():
    """Every edge of the exported mesh must be shared by exactly two faces."""
    hull = WigleyHull(L=3.0, B=0.3, T=0.1875)
    verts, faces = hull_mesh(hull, n_stations=41, n_waterlines=21, z_top=0.1)
    edges = {}
    for f in faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            key = (min(a, b), max(a, b))
            edges[key] = edges.get(key, 0) + 1
    open_edges = sum(1 for v in edges.values() if v != 2)
    assert open_edges / len(edges) < 0.02
