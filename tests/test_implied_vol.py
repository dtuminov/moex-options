"""Tests for solve_implied_vol.

The central test is a round-trip: pick a true vol, price it with Black-76,
then solve for implied vol from that price and confirm the original vol
comes back. That's the actual "compare model to market" correctness check —
if pricing and inversion don't round-trip, nothing downstream (the vol
surface) can be trusted.
"""

from __future__ import annotations

import pytest

from moex_options.black76 import OptionType, price
from moex_options.implied_vol import ImpliedVolError, solve_implied_vol


@pytest.mark.parametrize(
    "option_type,forward,strike,maturity,rate,true_vol",
    [
        (OptionType.CALL, 100.0, 100.0, 1.0, 0.05, 0.20),
        (OptionType.PUT, 100.0, 100.0, 1.0, 0.05, 0.20),
        (OptionType.CALL, 120.0, 100.0, 0.25, 0.03, 0.45),  # deep ITM call, short-dated
        (OptionType.PUT, 80.0, 100.0, 0.25, 0.03, 0.45),  # deep ITM put, short-dated
        (OptionType.CALL, 100.0, 150.0, 0.5, 0.10, 0.60),  # far OTM call, high vol
        (OptionType.CALL, 100.0, 100.0, 2.0, 0.0, 0.10),  # zero rate, long-dated, low vol
    ],
)
def test_round_trip_recovers_the_true_vol(
    option_type: OptionType,
    forward: float,
    strike: float,
    maturity: float,
    rate: float,
    true_vol: float,
) -> None:
    market_price = price(option_type, forward, strike, maturity, rate, true_vol)

    recovered_vol = solve_implied_vol(option_type, market_price, forward, strike, maturity, rate)

    assert recovered_vol == pytest.approx(true_vol, abs=1e-6)


def test_far_otm_short_dated_case_where_newton_alone_tends_to_struggle() -> None:
    # Vega is tiny here (far OTM, short-dated) -> plain Newton is exactly
    # the regime that stalls or overshoots, which is what the bisection
    # fallback exists for.
    option_type, forward, strike, maturity, rate, true_vol = (
        OptionType.CALL, 100.0, 200.0, 0.05, 0.02, 0.80,
    )  # fmt: skip
    market_price = price(option_type, forward, strike, maturity, rate, true_vol)

    recovered_vol = solve_implied_vol(option_type, market_price, forward, strike, maturity, rate)

    assert recovered_vol == pytest.approx(true_vol, abs=1e-4)


def test_price_above_upper_no_arbitrage_bound_raises() -> None:
    # Call upper bound is discount*forward; ask for more than that.
    with pytest.raises(ImpliedVolError, match="no-arbitrage"):
        solve_implied_vol(
            OptionType.CALL,
            market_price=1000.0,
            forward=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
        )


def test_price_below_lower_no_arbitrage_bound_raises() -> None:
    # Call lower bound is discount*max(F-K, 0) = discount*20 here; ask for less.
    with pytest.raises(ImpliedVolError, match="no-arbitrage"):
        solve_implied_vol(
            OptionType.CALL, market_price=1.0, forward=120.0, strike=100.0, maturity=1.0, rate=0.05
        )
