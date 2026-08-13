"""Tests for Black-76 pricing and Greeks.

Where possible these check exact mathematical identities (put-call parity,
delta_call - delta_put = discount, gamma/vega symmetry) rather than
eyeballed plausibility, and cross-check one reference price against an
independently computed normal CDF (Python's own `math.erf`, not scipy's
`norm.cdf` — so the test isn't just calling the same code twice).
"""

from __future__ import annotations

import math

import pytest

from moex_options.black76 import OptionType, greeks, price


def _independent_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def test_atm_call_matches_independently_computed_reference_value() -> None:
    forward, strike, maturity, rate, vol = 100.0, 100.0, 1.0, 0.05, 0.2

    d1 = (math.log(forward / strike) + 0.5 * vol**2 * maturity) / (vol * math.sqrt(maturity))
    d2 = d1 - vol * math.sqrt(maturity)
    discount = math.exp(-rate * maturity)
    expected = discount * (
        forward * _independent_normal_cdf(d1) - strike * _independent_normal_cdf(d2)
    )

    result = price(OptionType.CALL, forward, strike, maturity, rate, vol)

    assert result == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize(
    "forward,strike,maturity,rate,vol",
    [
        (100.0, 100.0, 1.0, 0.05, 0.2),
        (120.0, 100.0, 0.5, 0.03, 0.35),
        (80.0, 100.0, 2.0, 0.10, 0.15),
        (100.0, 100.0, 0.01, 0.0, 0.5),
    ],
)
def test_put_call_parity(
    forward: float, strike: float, maturity: float, rate: float, vol: float
) -> None:
    call = price(OptionType.CALL, forward, strike, maturity, rate, vol)
    put = price(OptionType.PUT, forward, strike, maturity, rate, vol)
    discount = math.exp(-rate * maturity)

    assert call - put == pytest.approx(discount * (forward - strike), abs=1e-9)


def test_call_price_approaches_discounted_intrinsic_value_as_vol_shrinks() -> None:
    forward, strike, maturity, rate = 120.0, 100.0, 1.0, 0.05

    result = price(OptionType.CALL, forward, strike, maturity, rate, vol=1e-4)

    discount = math.exp(-rate * maturity)
    assert result == pytest.approx(discount * (forward - strike), abs=1e-2)


def test_delta_call_minus_delta_put_equals_discount_factor() -> None:
    forward, strike, maturity, rate, vol = 100.0, 90.0, 0.75, 0.04, 0.25

    call_greeks = greeks(OptionType.CALL, forward, strike, maturity, rate, vol)
    put_greeks = greeks(OptionType.PUT, forward, strike, maturity, rate, vol)

    assert call_greeks.delta - put_greeks.delta == pytest.approx(math.exp(-rate * maturity))


def test_gamma_and_vega_are_identical_for_call_and_put() -> None:
    forward, strike, maturity, rate, vol = 100.0, 110.0, 0.5, 0.02, 0.3

    call_greeks = greeks(OptionType.CALL, forward, strike, maturity, rate, vol)
    put_greeks = greeks(OptionType.PUT, forward, strike, maturity, rate, vol)

    assert call_greeks.gamma == pytest.approx(put_greeks.gamma)
    assert call_greeks.vega == pytest.approx(put_greeks.vega)
    assert call_greeks.gamma > 0
    assert call_greeks.vega > 0


def test_call_delta_is_bounded_by_the_discount_factor() -> None:
    forward, strike, maturity, rate, vol = 100.0, 100.0, 1.0, 0.05, 0.2

    result = greeks(OptionType.CALL, forward, strike, maturity, rate, vol)

    discount = math.exp(-rate * maturity)
    assert 0.0 < result.delta < discount


@pytest.mark.parametrize(
    "forward,strike,maturity,rate,vol",
    [
        (100.0, 100.0, 1.0, 0.05, 0.2),
        (120.0, 100.0, 0.5, 0.03, 0.35),
        (80.0, 100.0, 2.0, 0.10, 0.15),
    ],
)
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_theta_matches_a_central_finite_difference_of_price_in_maturity(
    option_type: OptionType, forward: float, strike: float, maturity: float, rate: float, vol: float
) -> None:
    # theta is dPrice/dt (calendar time), i.e. -dPrice/dT (time-to-maturity)
    # -- price loses time value as maturity shrinks.
    h = 1e-5
    price_plus = price(option_type, forward, strike, maturity + h, rate, vol)
    price_minus = price(option_type, forward, strike, maturity - h, rate, vol)
    finite_diff_theta = -(price_plus - price_minus) / (2 * h)

    result = greeks(option_type, forward, strike, maturity, rate, vol)

    assert result.theta == pytest.approx(finite_diff_theta, abs=1e-3)


@pytest.mark.parametrize(
    "forward,strike,maturity,rate,vol",
    [
        (100.0, 100.0, 1.0, 0.05, 0.2),
        (120.0, 100.0, 0.5, 0.03, 0.35),
        (80.0, 100.0, 2.0, 0.10, 0.15),
    ],
)
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_rho_matches_a_central_finite_difference_of_price_in_rate(
    option_type: OptionType, forward: float, strike: float, maturity: float, rate: float, vol: float
) -> None:
    h = 1e-6
    price_plus = price(option_type, forward, strike, maturity, rate + h, vol)
    price_minus = price(option_type, forward, strike, maturity, rate - h, vol)
    finite_diff_rho = (price_plus - price_minus) / (2 * h)

    result = greeks(option_type, forward, strike, maturity, rate, vol)

    assert result.rho == pytest.approx(finite_diff_rho, abs=1e-3)


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"forward": 0.0},
        {"forward": -1.0},
        {"strike": 0.0},
        {"maturity": 0.0},
        {"maturity": -1.0},
        {"vol": 0.0},
        {"vol": -0.1},
    ],
)
def test_price_rejects_non_positive_inputs(bad_kwargs: dict[str, float]) -> None:
    kwargs = {"forward": 100.0, "strike": 100.0, "maturity": 1.0, "rate": 0.05, "vol": 0.2}
    kwargs.update(bad_kwargs)
    with pytest.raises(ValueError):
        price(OptionType.CALL, **kwargs)
