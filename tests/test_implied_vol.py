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
    # the regime that stalls or overshoots, which is what the Brent
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


def test_no_arbitrage_violation_has_a_distinct_reason_from_no_solution_in_range() -> None:
    # Two different ImpliedVolError causes should be programmatically
    # distinguishable via `reason`, not just by eyeballing the message: an
    # actually-invalid quote (violates a no-arbitrage bound) vs. a valid,
    # arbitrage-free quote for which the solver's search range just doesn't
    # contain a matching vol.
    with pytest.raises(ImpliedVolError) as exc_info:
        solve_implied_vol(
            OptionType.CALL,
            market_price=1000.0,
            forward=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
        )
    assert exc_info.value.reason == "no_arbitrage_violation"


def test_non_positive_maturity_raises_implied_vol_error_not_bare_value_error() -> None:
    # solve_implied_vol must validate maturity itself and raise its own
    # ImpliedVolError, rather than letting a bare ValueError bubble up from
    # black76._d1_d2 deep in the call stack -- callers of this module should
    # get one consistent exception type regardless of which invalid input
    # triggered it.
    with pytest.raises(ImpliedVolError) as exc_info:
        solve_implied_vol(
            OptionType.CALL, market_price=10.0, forward=100.0, strike=100.0, maturity=0.0, rate=0.05
        )
    assert exc_info.value.reason == "invalid_maturity"

    with pytest.raises(ImpliedVolError):
        solve_implied_vol(
            OptionType.CALL,
            market_price=10.0,
            forward=100.0,
            strike=100.0,
            maturity=-1.0,
            rate=0.05,
        )


@pytest.mark.parametrize("initial_guess", [0.05, 0.3, 0.9, 2.0])
def test_deep_otm_short_dated_zero_price_never_returns_the_unmodified_initial_guess(
    initial_guess: float,
) -> None:
    # Regression test for the false-convergence bug: F=100, K=1000, T=0.01
    # is a far-OTM, short-dated call whose Black-76 price underflows to
    # exactly 0.0 in float64. The old code's very first convergence check
    # (`abs(diff) < tol`) trivially passed against a market_price of 0.0
    # before a single real Newton step was taken -- for every one of these
    # four initial guesses, "recovering" the unmodified initial guess as if
    # it were a real solved vol, with no error and no signal anything was
    # wrong. The fix must never do that: it should raise a clear error
    # instead (since a 0.0 quote carries no vol information at all).
    with pytest.raises(ImpliedVolError) as exc_info:
        solve_implied_vol(
            OptionType.CALL,
            market_price=0.0,
            forward=100.0,
            strike=1000.0,
            maturity=0.01,
            rate=0.0,
            initial_guess=initial_guess,
        )
    assert exc_info.value.reason == "zero_price"


@pytest.mark.parametrize("initial_guess", [0.05, 0.3, 0.9, 2.0])
def test_near_zero_but_nonzero_price_still_does_not_return_the_unmodified_initial_guess(
    initial_guess: float,
) -> None:
    # Same regime as the zero-price regression above, but with a tiny
    # nonzero market_price that bypasses the explicit zero_price check --
    # this exercises the vega-gated convergence check instead (fix #2:
    # `diff` alone is not trusted as "converged" unless vega confirms the
    # solver is in a regime where a step is meaningful). Should still never
    # trivially return the untouched initial guess.
    recovered_vol = solve_implied_vol(
        OptionType.CALL,
        market_price=1e-150,
        forward=100.0,
        strike=1000.0,
        maturity=0.01,
        rate=0.0,
        initial_guess=initial_guess,
    )
    assert recovered_vol != initial_guess
