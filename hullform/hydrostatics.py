"""Hydrostatics computed from the offset field, by direct integration.

Nothing here is read from a table: displacement, centres, the curves of form
and the righting levers are all quadratures over the half-breadth field, so
the same routine works for any hull in geometry.py.

Integration is Simpson's rule on an odd number of evenly spaced ordinates,
which is what a ship's hydrostatics has been done with since long before
computers, and which converges fast enough that the Wigley hull's exact
coefficients come back to seven figures.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

RHO_SW = 1025.0  # density of salt water, kg/m^3
G = 9.80665


def simpson(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Composite Simpson's rule over evenly spaced x (odd number of points)."""
    n = y.shape[axis]
    if n < 3 or n % 2 == 0:
        raise ValueError(f"Simpson needs an odd number of ordinates >= 3, got {n}")
    h = (x[-1] - x[0]) / (n - 1)
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    shape = [1] * y.ndim
    shape[axis] = n
    return (h / 3.0) * np.sum(y * w.reshape(shape), axis=axis)


@dataclass
class Hydrostatics:
    """Hydrostatic particulars at one draught."""

    draught: float
    volume: float          # moulded displaced volume, m^3
    displacement: float    # mass displacement in salt water, tonnes
    Aw: float              # waterplane area, m^2
    Am: float              # midship section area, m^2
    LCB: float             # longitudinal centre of buoyancy from midships, m (+fwd)
    LCF: float             # longitudinal centre of flotation from midships, m (+fwd)
    KB: float              # vertical centre of buoyancy above keel, m
    BMt: float             # transverse metacentric radius, m
    BMl: float             # longitudinal metacentric radius, m
    KMt: float             # transverse metacentre above keel, m
    Cb: float              # block coefficient
    Cw: float              # waterplane area coefficient
    Cm: float              # midship section coefficient
    Cp: float              # prismatic coefficient
    TPC: float             # tonnes per cm immersion
    MCT1cm: float          # moment to change trim one cm, tonne-m

    def as_dict(self):
        return asdict(self)


class HydrostaticSolver:
    """Integrates a hull's offsets into hydrostatic particulars.

    n_stations and n_waterlines must both be odd (Simpson's rule).  The
    defaults are far finer than a drawing office would use, because the point
    here is to show convergence to the analytic answer.
    """

    def __init__(self, hull, n_stations: int = 201, n_waterlines: int = 201):
        if n_stations % 2 == 0 or n_waterlines % 2 == 0:
            raise ValueError("station and waterline counts must be odd")
        self.hull = hull
        self.ns = n_stations
        self.nw = n_waterlines

    # -- primary quadratures ---------------------------------------------

    def at_draught(self, T: float | None = None) -> Hydrostatics:
        hull = self.hull
        T = hull.T if T is None else float(T)

        x = np.linspace(-hull.L / 2, hull.L / 2, self.ns)
        z = np.linspace(-hull.T, -hull.T + T, self.nw)  # keel upwards
        y = hull.halfbreadth(x[:, None], z[None, :])  # (station, waterline)

        # sectional areas A(x) and the volume
        A = 2.0 * simpson(y, z, axis=1)                    # m^2 per station
        volume = simpson(A, x, axis=0)                     # m^3

        # waterplane at the top of the range
        yw = y[:, -1]
        Aw = 2.0 * simpson(yw, x, axis=0)

        # centres
        LCB = simpson(A * x, x, axis=0) / volume
        LCF = 2.0 * simpson(yw * x, x, axis=0) / Aw
        # KB: moment of volume about the keel
        zk = z - z[0]                                       # height above keel
        moment_z = simpson(2.0 * simpson(y * zk[None, :], z, axis=1), x, axis=0)
        KB = moment_z / volume

        # second moments of the waterplane
        I_T = (2.0 / 3.0) * simpson(yw**3, x, axis=0)
        I_L_origin = 2.0 * simpson(yw * x**2, x, axis=0)
        I_L = I_L_origin - Aw * LCF**2                       # about the LCF

        Am = A.max()
        Cb = volume / (hull.L * hull.B * T)
        Cw = Aw / (hull.L * hull.B)
        Cm = Am / (hull.B * T)
        Cp = volume / (Am * hull.L)

        TPC = RHO_SW * Aw / 1000.0 / 100.0                   # tonnes per cm
        BMl = I_L / volume
        MCT1cm = (RHO_SW * volume / 1000.0) * BMl / (100.0 * hull.L)

        return Hydrostatics(
            draught=T,
            volume=volume,
            displacement=RHO_SW * volume / 1000.0,
            Aw=Aw,
            Am=Am,
            LCB=LCB,
            LCF=LCF,
            KB=KB,
            BMt=I_T / volume,
            BMl=BMl,
            KMt=KB + I_T / volume,
            Cb=Cb,
            Cw=Cw,
            Cm=Cm,
            Cp=Cp,
            TPC=TPC,
            MCT1cm=MCT1cm,
        )

    # -- derived curves ---------------------------------------------------

    def curves_of_form(self, n: int = 21):
        """Hydrostatics at n draughts from 10% to 100% of the design draught."""
        draughts = np.linspace(0.1 * self.hull.T, self.hull.T, n)
        return draughts, [self.at_draught(t) for t in draughts]

    def bonjean(self, n_stations: int = 21, n_draughts: int = 21):
        """Bonjean curves: sectional area against draught, per station.

        These are what a ship is trimmed and floated with in practice: read
        the immersed area at each station for a given waterline, integrate,
        and you have the displacement of any floating condition.
        """
        hull = self.hull
        x = np.linspace(-hull.L / 2, hull.L / 2, n_stations)
        draughts = np.linspace(hull.T / n_draughts, hull.T, n_draughts)
        A = np.zeros((n_stations, n_draughts))
        zfine = np.linspace(-hull.T, 0.0, self.nw)
        for j, t in enumerate(draughts):
            zz = np.linspace(-hull.T, -hull.T + t, self.nw)
            yy = hull.halfbreadth(x[:, None], zz[None, :])
            A[:, j] = 2.0 * simpson(yy, zz, axis=1)
        return x, draughts, A

    # -- stability --------------------------------------------------------

    def gz_curve(
        self,
        KG: float,
        heels_deg=np.arange(0.0, 61.0, 5.0),
        n_grid: int = 161,
        n_sections: int = 61,
    ):
        """Righting lever GZ(phi) by direct integration of the immersed volume.

        For each heel angle the waterline is bisected until the immersed
        volume matches the upright displacement (free trim is not modelled),
        and the immersed cells give the centre of buoyancy.  GZ then follows
        from GZ = y_B cos(phi) + (z_B - KG) sin(phi).

        The hull interior is voxelised once, outside the heel loop: the mask
        does not move with the ship, only the waterline plane does.
        """
        hull = self.hull
        target_V = self.at_draught().volume

        ns = n_sections if n_sections % 2 else n_sections + 1
        x = np.linspace(-hull.L / 2, hull.L / 2, ns)
        yy = np.linspace(-hull.B, hull.B, n_grid)
        zz = np.linspace(-hull.T, hull.T, n_grid)
        dA = (yy[1] - yy[0]) * (zz[1] - zz[0])

        Y, Z = np.meshgrid(yy, zz, indexing="ij")
        yb = hull.halfbreadth(x[:, None, None], Z[None, :, :])
        inside = np.abs(Y)[None, :, :] <= yb                    # (station, y, z)
        st_idx, yi, zi = np.nonzero(inside)
        cy, cz = Y[yi, zi], Z[yi, zi]
        counts_shape = ns

        def condition(phi, zw):
            """Immersed volume and moments for this heel and waterline."""
            # a ship-fixed point (y, z) sits at global height z cos(phi) - y sin(phi)
            wet = (cz * np.cos(phi) - cy * np.sin(phi)) <= zw
            idx = st_idx[wet]
            area = np.bincount(idx, minlength=counts_shape) * dA
            my = np.bincount(idx, weights=cy[wet], minlength=counts_shape) * dA
            mz = np.bincount(idx, weights=cz[wet], minlength=counts_shape) * dA
            return (simpson(area, x), simpson(my, x), simpson(mz, x))

        gz = np.zeros(len(heels_deg))
        for i, phi_deg in enumerate(np.asarray(heels_deg, dtype=float)):
            phi = np.radians(phi_deg)
            lo, hi = -hull.T, hull.T
            for _ in range(40):
                zw = 0.5 * (lo + hi)
                V, my, mz = condition(phi, zw)
                if V < target_V:
                    lo = zw
                else:
                    hi = zw
            V, my, mz = condition(phi, 0.5 * (lo + hi))
            # KG is measured from the keel, so put the buoyancy centre on the
            # same datum: the grid has z = 0 at the design waterline.
            yB, zB = my / V, mz / V + hull.T
            gz[i] = yB * np.cos(phi) + (zB - KG) * np.sin(phi)
        return np.asarray(heels_deg, dtype=float), gz
