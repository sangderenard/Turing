"""Authoring/units tests only. Native compiler execution has a separate probe."""
import math
import pytest
import sympy as sp
from examples.meowdoku.studio_filter_density import (
    U,V,bezier_density,neutral_density_sigma,sigma_from_extinction,straight_filter_path,_validate_lambda,
)


def test_bicubic_is_actual_bernstein_polynomial():
    field=bezier_density();controls=field.expr.free_symbols-{U,V}
    assert len(controls)==16
    assert sp.simplify(field.expr.subs({x:1 for x in controls}))==1
    assert sp.simplify(field.expr.subs({sp.Symbol(f'density_{i}_{j}'):sp.Rational(i,3) for i in range(4) for j in range(4)}))==U
    assert field.expr.subs({U:0,V:0})==sp.Symbol('density_0_0')
    assert field.expr.subs({U:1,V:1})==sp.Symbol('density_3_3')


def test_nd_means_base10_power_density_not_opacity():
    for nd in (0,.3,1,2):
        sigma=neutral_density_sigma(nd,.001)
        assert math.exp(-sigma*.001)==pytest.approx(10**(-nd))
        for wavelength in (.4,.55,.7):
            k=sigma*wavelength*1e-6/(4*math.pi)
            assert sigma_from_extinction(k,wavelength)==pytest.approx(sigma)


def test_varying_oblique_path_cannot_use_entry_density():
    assert straight_filter_path(.001,1)==.001
    assert straight_filter_path(.001,-.5,spatially_uniform=True)==.002
    with pytest.raises(ValueError,match='path integral'):straight_filter_path(.001,.5)
    with pytest.raises(ValueError):straight_filter_path(.001,0,spatially_uniform=True)


def test_unsafe_or_unbounded_authoring_refuses():
    with pytest.raises(ValueError):bezier_density(4)
    with pytest.raises(ValueError):_validate_lambda('lambda u,v: 1')
    with pytest.raises(ValueError):_validate_lambda(sp.Lambda((U,V),sp.Symbol('sigma_a_per_m')))
    with pytest.raises(ValueError):neutral_density_sigma(-1,.001)
    with pytest.raises(ValueError):sigma_from_extinction(1,0)

