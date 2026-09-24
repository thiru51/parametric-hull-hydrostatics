"""Parametric hull surfaces, defined by form parameters.

A hull here is a callable half-breadth field y(x, z), plus the principal
dimensions it was built from.  Everything downstream -- hydrostatics, the
Rhino model, the CFD mesh -- reads that one field, so the geometry is defined
in exactly one place.

Coordinates, throughout the project:

    x   along the ship, 0 at midships, +x forward, range [-L/2, +L/2]
    y   half-breadth (offset), y >= 0, the hull is mirrored about y = 0
    z   vertical, 0 at the design waterline, +z up, keel at z = -T
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Hull:
    """A hull form: principal dimensions plus a half-breadth field."""

    name: str
    L: float  # length on the waterline, m
    B: float  # moulded breadth, m
    T: float  # design draught, m

    def halfbreadth(self, x, z):
        """Half-breadth y(x, z), zero outside the hull."""
        raise NotImplementedError

    # -- sampling helpers ------------------------------------------------

    def stations(self, n: int) -> np.ndarray:
        """n station positions from aft perpendicular to forward."""
        return np.linspace(-self.L / 2, self.L / 2, n)

    def waterlines(self, n: int, zmax: float | None = None) -> np.ndarray:
        """n waterline heights from the keel up to zmax (default: draught)."""
        return np.linspace(-self.T, 0.0 if zmax is None else zmax, n)

    def offsets(self, n_stations: int = 21, n_waterlines: int = 21):
        """The offset table: (x, z, y[station, waterline])."""
        x = self.stations(n_stations)
        z = self.waterlines(n_waterlines)
        y = self.halfbreadth(x[:, None], z[None, :])
        return x, z, y


class WigleyHull(Hull):
    """The Wigley parabolic hull.

    y(x, z) = (B/2) (1 - (2x/L)^2) (1 - (z/T)^2)

    It is the standard analytic test hull: its coefficients are exact
    rational numbers (Cb = 4/9, Cw = Cm = Cp = 2/3), which makes it the right
    thing to validate a hydrostatics routine against.
    """

    def __init__(self, L: float = 100.0, B: float = 10.0, T: float = 6.25):
        super().__init__(name="wigley", L=L, B=B, T=T)

    def halfbreadth(self, x, z):
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        xi = 2.0 * x / self.L
        zeta = z / self.T
        y = (self.B / 2.0) * (1.0 - xi**2) * (1.0 - zeta**2)
        inside = (np.abs(xi) <= 1.0) & (z >= -self.T)
        return np.where(inside, np.maximum(y, 0.0), 0.0)

    # exact values, for the validation suite
    exact = {"Cb": 4.0 / 9.0, "Cw": 2.0 / 3.0, "Cm": 2.0 / 3.0, "Cp": 2.0 / 3.0}

    def exact_KB(self) -> float:
        """Vertical centre of buoyancy above the keel: 5T/8."""
        return 5.0 * self.T / 8.0

    def exact_BMt(self) -> float:
        """Transverse metacentric radius I_T / V, in closed form."""
        I_T = (2.0 / 3.0) * (self.B / 2.0) ** 3 * (16.0 * self.L / 35.0)
        V = (4.0 / 9.0) * self.L * self.B * self.T
        return I_T / V

    def exact_LCB(self) -> float:
        """Longitudinal centre of buoyancy: amidships, by symmetry."""
        return 0.0


class SeriesHull(Hull):
    """A parametric ship form built from three shape parameters.

    Unlike the Wigley hull this one looks like a ship: it has a bulb-free but
    full forebody, parallel middle body, and a fined run aft.  The section
    shape is a power-law ("Lackenby-like") family whose exponent sets the
    bilge fullness, and the sectional area curve sets the prismatic
    coefficient.

        n_sec   section exponent; 1 = wall-sided V, large = box-like U
        pmb     parallel middle body as a fraction of L
        Cp_aim  prismatic coefficient the area curve is stretched towards
    """

    def __init__(
        self,
        L: float = 100.0,
        B: float = 16.0,
        T: float = 6.0,
        n_sec: float = 2.2,
        pmb: float = 0.18,
        Cp_aim: float = 0.62,
        name: str = "series",
    ):
        super().__init__(name=name, L=L, B=B, T=T)
        object.__setattr__(self, "n_sec", float(n_sec))
        object.__setattr__(self, "pmb", float(pmb))
        object.__setattr__(self, "Cp_aim", float(Cp_aim))

    # -- shape functions -------------------------------------------------

    def _area_shape(self, xi):
        """Sectional area curve, normalised to 1 amidships, over xi in [-1, 1].

        Flat over the parallel middle body, then falling to zero at the ends;
        the fall is sharper aft than forward, as on a real ship.
        """
        xi = np.asarray(xi, dtype=float)
        h = self.pmb  # half-width of the parallel middle body, in xi
        t = (np.abs(xi) - h) / (1.0 - h)
        t = np.clip(t, 0.0, 1.0)
        # exponent > 1 forward (fuller entry), < 1 aft (finer run)
        p = np.where(xi >= 0.0, 1.0 + 2.0 * (1.0 - self.Cp_aim), 1.0 + 1.2 * (1.0 - self.Cp_aim))
        s = 1.0 - t**p
        return np.where(np.abs(xi) <= 1.0, np.clip(s, 0.0, 1.0), 0.0)

    def _section_shape(self, zeta):
        """Girth distribution with depth: 1 at the waterline, 0 at the keel."""
        zeta = np.asarray(zeta, dtype=float)  # 0 at keel, 1 at waterline
        return np.clip(zeta, 0.0, 1.0) ** (1.0 / self.n_sec)

    def halfbreadth(self, x, z):
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        xi = 2.0 * x / self.L
        zeta = 1.0 + z / self.T  # 0 at keel, 1 at DWL
        y = (self.B / 2.0) * self._area_shape(xi) * self._section_shape(zeta)
        inside = (np.abs(xi) <= 1.0) & (z >= -self.T)
        return np.where(inside, np.maximum(y, 0.0), 0.0)


def build(kind: str, **kwargs) -> Hull:
    """Factory used by the CLI: build('wigley', L=..., B=..., T=...)."""
    kinds = {"wigley": WigleyHull, "series": SeriesHull}
    if kind not in kinds:
        raise ValueError(f"unknown hull '{kind}', expected one of {sorted(kinds)}")
    return kinds[kind](**kwargs)
