"""Implied volatility: inverts Black-76 by numerical root-finding.

Newton-Raphson (fast, using vega as the derivative) first, falling back to
Brent's method (`scipy.optimize.brentq` — slower, but guaranteed to converge
given a bracketing interval) when Newton stalls or steps outside a sane vol
range. Plain Newton diverges or oscillates for far-OTM or short-dated
options, where price is nearly flat in vol (vega close to 0) and a single
step overshoots.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from moex_options.black76 import OptionType, greeks, price

_MAX_NEWTON_ITER = 50
_NEWTON_PRICE_TOL = 1e-8
_MIN_VEGA = 1e-8
_VOL_LOWER_BOUND = 1e-4
_VOL_UPPER_BOUND = 5.0


class ImpliedVolError(RuntimeError):
    """No implied vol could be found — typically because `market_price`
    violates a no-arbitrage bound for the given forward/strike/maturity."""


def solve_implied_vol(
    option_type: OptionType,
    market_price: float,
    forward: float,
    strike: float,
    maturity: float,
    rate: float,
    initial_guess: float = 0.3,
) -> float:
    _check_no_arbitrage_bound(option_type, market_price, forward, strike, maturity, rate)

    vol = initial_guess
    for _ in range(_MAX_NEWTON_ITER):
        diff = price(option_type, forward, strike, maturity, rate, vol) - market_price
        if abs(diff) < _NEWTON_PRICE_TOL:
            return vol
        vega = greeks(option_type, forward, strike, maturity, rate, vol).vega
        if vega < _MIN_VEGA:
            break
        vol = vol - diff / vega
        if not (_VOL_LOWER_BOUND < vol < _VOL_UPPER_BOUND):
            break

    return _solve_by_brent(option_type, market_price, forward, strike, maturity, rate)


def _solve_by_brent(
    option_type: OptionType,
    market_price: float,
    forward: float,
    strike: float,
    maturity: float,
    rate: float,
) -> float:
    def objective(vol: float) -> float:
        return price(option_type, forward, strike, maturity, rate, vol) - market_price

    lo, hi = _VOL_LOWER_BOUND, _VOL_UPPER_BOUND
    if objective(lo) * objective(hi) > 0:
        raise ImpliedVolError(
            f"no solution in vol range [{lo}, {hi}] for market_price={market_price}"
        )
    return float(brentq(objective, lo, hi, xtol=1e-10))


def _check_no_arbitrage_bound(
    option_type: OptionType,
    market_price: float,
    forward: float,
    strike: float,
    maturity: float,
    rate: float,
) -> None:
    discount = float(np.exp(-rate * maturity))
    if option_type is OptionType.CALL:
        lower_bound = discount * max(forward - strike, 0.0)
        upper_bound = discount * forward
    else:
        lower_bound = discount * max(strike - forward, 0.0)
        upper_bound = discount * strike

    if not (lower_bound - 1e-9 <= market_price <= upper_bound + 1e-9):
        raise ImpliedVolError(
            f"market_price={market_price} violates no-arbitrage bounds "
            f"[{lower_bound:.6f}, {upper_bound:.6f}]"
        )
