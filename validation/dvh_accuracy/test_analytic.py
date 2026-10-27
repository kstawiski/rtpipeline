"""Independent integration and symmetry checks for the continuous truth."""
import math
import numpy as np
import pytest
from scipy.integrate import quad
from analytic import Geometry, ball_below, dose_at_fraction, extrema, metrics, survival, z_cdf


@pytest.mark.parametrize("shape", ["sphere", "shell", "cube", "cylinder"])
@pytest.mark.parametrize("volume", [2., 50., 1000.])
def test_geometry_and_linear_against_cross_section_integrals(shape, volume):
    g = Geometry(shape, volume, (.7, -.4, .3))
    r, a = g.radius, g.inner_radius
    def area(z):
        if shape == "cube":
            return (2 * r) ** 2
        if shape == "cylinder":
            return math.pi * r ** 2
        return math.pi * (r * r - z * z - max(0., a * a - z * z))
    points = [-a, a] if a else None
    v = quad(area, -r, r, points=points, epsabs=1e-6)[0]
    mean = quad(lambda z: area(z) * (20 + .1 * (z + .3)), -r, r,
                points=points, epsabs=1e-5)[0] / v
    assert v / 1000 == pytest.approx(volume, rel=1e-12)
    assert mean == pytest.approx(metrics(g, "linear")["Dmean_Gy"], abs=1e-10)
    for z in (-.83 * r, -.2 * r, .37 * r, .91 * r):
        numerical = quad(area, -r, z, points=[q for q in (points or []) if -r < q < z],
                         epsabs=1e-6)[0] / v
        assert z_cdf(g, z + .3) == pytest.approx(numerical, abs=1e-10)
    for p in (.00003, .02, .5, .95, .98):
        d = dose_at_fraction(g, "linear", p)
        assert survival(g, "linear", d) == pytest.approx(p, abs=1e-12)


@pytest.mark.parametrize("shape", ["sphere", "shell"])
@pytest.mark.parametrize("volume", [2., 50., 1000.])
def test_radial_mean_and_quantiles_against_integration(shape, volume):
    g = Geometry(shape, volume)
    r, a = g.radius, g.inner_radius
    integ = quad(lambda x: (30 - .15 * x) * 4 * math.pi * x * x, a, r)[0] / (volume * 1000)
    assert integ == pytest.approx(metrics(g, "radial")["Dmean_Gy"], abs=1e-10)
    for p in (.00003, .02, .5, .95, .98, 1.):
        d = dose_at_fraction(g, "radial", p)
        assert survival(g, "radial", d) == pytest.approx(p, abs=1e-12)


def test_uniform_atom_volume_boundaries_and_units():
    g = Geometry("sphere", 2.)
    m = metrics(g, "uniform")
    assert m["V20Gy_cc"] == 2
    assert m["V95Rx_pct"] == 0  # 95% of 25 Gy = 23.75 Gy
    assert survival(g, "uniform", np.nextafter(20., math.inf)) == 0
    assert all(m[k] == 20 for k in m if k.startswith("D"))
    assert metrics(Geometry("sphere", 1.999), "linear")["D2cc_Gy"] is None
    assert metrics(Geometry("sphere", .029), "linear")["D0.03cc_Gy"] is None
    assert metrics(Geometry("cube", 1000), "linear")["V20Gy_cc"] == 500
    assert metrics(Geometry("cube", 1000), "linear")["D95_Gy"] == pytest.approx(15.5)
    assert metrics(Geometry("sphere", 2), "linear")["D2cc_Gy"] == extrema(g, "linear")[0]


def test_saturation_symmetry_and_invalid_cases():
    for shape in ("sphere", "shell", "cube", "cylinder"):
        g = Geometry(shape, 50)
        assert z_cdf(g, -2 * g.radius) == 0
        assert z_cdf(g, 2 * g.radius) == 1
        assert z_cdf(g, 0) == pytest.approx(.5)
        for z in np.linspace(0, g.radius, 11):
            assert z_cdf(g, z) + z_cdf(g, -z) == pytest.approx(1)
    assert ball_below(1, 0) == 0
    for args in (("wrong", 2), ("sphere", 0)):
        with pytest.raises(ValueError):
            Geometry(*args)
    with pytest.raises(ValueError):
        dose_at_fraction(Geometry("sphere", 2), "linear", 1.01)
    with pytest.raises(ValueError):
        survival(Geometry("cube", 2), "radial", 20)
