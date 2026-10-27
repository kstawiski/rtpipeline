"""Continuous geometric truth, in mm, cm3 and Gy; no voxel or contour input.

The cumulative DVH is V(D >= d). Dp is its generalized inverse at p/100.
The linear field is 20 + 0.1*z Gy. The radial field is 30 - 0.15*r Gy,
centred on each sphere/shell. Uniform dose is 20 Gy. Reference dose is 25 Gy.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

REFERENCE_GY = 25.0
METRICS = ("Dmean_Gy", "Dmin_Gy", "Dmax_Gy", "D98_Gy", "D95_Gy", "D50_Gy",
           "D2_Gy", "D2cc_Gy", "D0.03cc_Gy", "V95Rx_pct", "V20Gy_cc", "Volume_cc")


@dataclass(frozen=True)
class Geometry:
    shape: str
    volume_cc: float
    center: tuple[float, float, float] = (0., 0., 0.)
    inner_ratio: float = 0.9

    def __post_init__(self):
        if self.shape not in {"sphere", "shell", "cylinder", "cube"}:
            raise ValueError("Unknown shape")
        if not self.volume_cc > 0 or not 0 < self.inner_ratio < 1:
            raise ValueError("Positive volume and 0 < inner_ratio < 1 required")

    @property
    def radius(self):
        v = self.volume_cc * 1000.
        if self.shape == "cube":
            return v ** (1 / 3) / 2  # half side length
        if self.shape == "cylinder":
            return (v / (2 * math.pi)) ** (1 / 3)  # height = diameter
        factor = 1 - self.inner_ratio ** 3 if self.shape == "shell" else 1.
        return (3 * v / (4 * math.pi * factor)) ** (1 / 3)

    @property
    def inner_radius(self):
        return self.radius * self.inner_ratio if self.shape == "shell" else 0.


def ball_below(z, radius):
    """Exact volume in mm3 below a plane at local z, including saturation."""
    if radius <= 0 or z <= -radius:
        return 0.
    if z >= radius:
        return 4 * math.pi * radius ** 3 / 3
    return math.pi * (radius * radius * z - z ** 3 / 3 + 2 * radius ** 3 / 3)


def z_cdf(g: Geometry, z):
    q, r = z - g.center[2], g.radius
    if q <= -r:
        return 0.
    if q >= r:
        return 1.
    if g.shape in {"cube", "cylinder"}:
        return min(1., max(0., (q + r) / (2 * r)))
    return min(1., max(0., (ball_below(q, r) - ball_below(q, g.inner_radius)) /
                          (g.volume_cc * 1000)))


def survival(g: Geometry, field: str, dose: float):
    """Exact fraction at or above dose; uniform-dose atom is inclusive."""
    if field == "uniform":
        return float(dose <= 20.)
    if field == "linear":
        return 1 - z_cdf(g, (dose - 20.) / .1)
    if field != "radial" or g.shape not in {"sphere", "shell"}:
        raise ValueError("Radial truth is defined only for concentric sphere/shell")
    r = min(g.radius, max(g.inner_radius, (30. - dose) / .15))
    return (r ** 3 - g.inner_radius ** 3) / (g.radius ** 3 - g.inner_radius ** 3)


def extrema(g: Geometry, field: str):
    if field == "uniform":
        return 20., 20.
    if field == "linear":
        return 20 + .1 * (g.center[2] - g.radius), 20 + .1 * (g.center[2] + g.radius)
    if field == "radial" and g.shape in {"sphere", "shell"}:
        return 30 - .15 * g.radius, 30 - .15 * g.inner_radius
    raise ValueError("Unsupported field")


def dose_at_fraction(g: Geometry, field: str, fraction: float):
    if not 0 <= fraction <= 1:
        raise ValueError("Fraction must lie in [0, 1]")
    lo, hi = extrema(g, field)
    if field == "uniform":
        return lo
    if field == "radial":
        r = (g.inner_radius ** 3 + fraction * (g.radius ** 3 - g.inner_radius ** 3)) ** (1 / 3)
        return 30 - .15 * r
    if fraction == 1:
        return lo
    if fraction == 0:
        return hi
    # Invert the closed-form piecewise polynomial CDF to < 1e-12 Gy.
    for _ in range(80):
        mid = (lo + hi) / 2
        if survival(g, field, mid) > fraction:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def metrics(g: Geometry, field: str):
    low, high = extrema(g, field)
    if field == "uniform":
        mean = 20.
    elif field == "linear":
        mean = 20 + .1 * g.center[2]
    else:
        r, a = g.radius, g.inner_radius
        mean = 30 - .15 * .75 * (r ** 4 - a ** 4) / (r ** 3 - a ** 3)
    out = {"Dmean_Gy": mean, "Dmin_Gy": low, "Dmax_Gy": high, "Volume_cc": g.volume_cc,
           "V95Rx_pct": 100 * survival(g, field, .95 * REFERENCE_GY),
           "V20Gy_cc": g.volume_cc * survival(g, field, 20.)}
    for p in (98, 95, 50, 2):
        out[f"D{p}_Gy"] = dose_at_fraction(g, field, p / 100)
    for cc in (2., .03):
        out[f"D{cc:g}cc_Gy"] = (dose_at_fraction(g, field, cc / g.volume_cc)
                                 if g.volume_cc >= cc else None)
    return out
