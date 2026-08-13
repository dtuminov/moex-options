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
# Both tolerances below are scaled to the instrument's own price/forward
# level rather than fixed absolute epsilons -- see `_price_tol`/`_min_vega`.
# An absolute 1e-8 price tolerance is effectively unreachable for a
# ~900 RUB MOEX premium (and trivially, wrongly, satisfied by a ~0 RUB one),
# and an absolute 1e-8 vega floor is far too small to act as a real
# safeguard at a ~29,000 RUB forward.
_NEWTON_PRICE_TOL_REL = 1e-8
_MIN_VEGA_FRACTION = 1e-6
_VOL_LOWER_BOUND = 1e-4
_VOL_UPPER_BOUND = 5.0


class ImpliedVolError(RuntimeError):
    """No implied vol could be found. `reason` distinguishes the (otherwise
    same-typed) failure modes programmatically:

    - "invalid_maturity": `maturity` was non-positive.
    - "no_arbitrage_violation": `market_price` itself violates a static
      no-arbitrage bound for the given forward/strike/maturity -- the quote
      is invalid regardless of any vol.
    - "zero_price": the price underflowed to exactly 0.0 in float64, so no
      vol can be recovered from it -- distinct from an arbitrage violation,
      since 0.0 is itself a valid (if uninformative) price.
    - "no_solution_in_range": `market_price` is a valid, arbitrage-free
      price, but no vol in the solver's search bracket
      [_VOL_LOWER_BOUND, _VOL_UPPER_BOUND] reproduces it.
    """

    def __init__(self, message: str, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


def _price_tol(market_price: float) -> float:
    """Newton price-convergence tolerance: a relative tolerance on the
    quoted price, floored at 1.0 so it stays sane as market_price -> 0
    (a literal `_NEWTON_PRICE_TOL_REL * 0` would make any nonzero diff
    fail to converge; more importantly, an exact market_price of 0.0 is
    handled explicitly before this is ever used -- see the zero_price
    check in `solve_implied_vol`).
    """
    return _NEWTON_PRICE_TOL_REL * max(abs(market_price), 1.0)


def _min_vega(forward: float) -> float:
    """Vega floor below which a Newton step is numerically meaningless,
    scaled to the forward level: vega has units of price-per-vol-point, so
    its natural scale tracks the instrument's own price level."""
    return _MIN_VEGA_FRACTION * forward


def solve_implied_vol(
    option_type: OptionType,
    market_price: float,
    forward: float,
    strike: float,
    maturity: float,
    rate: float,
    initial_guess: float = 0.3,
) -> float:
    if maturity <= 0:
        raise ImpliedVolError(
            f"maturity must be positive, got {maturity}", reason="invalid_maturity"
        )
    _check_no_arbitrage_bound(option_type, market_price, forward, strike, maturity, rate)

    if market_price == 0.0:
        # Both a real zero quote and a price that has underflowed to
        # exactly 0.0 in float64 (realistic for a deep-OTM, short-dated
        # contract) land here. Either way, no vol can be recovered: every
        # vol at or below the underflow threshold prices to the same 0.0,
        # so there's no unique root to find. Returning any particular vol
        # (e.g. the caller's initial guess, unmodified) would just fabricate
        # an answer.
        raise ImpliedVolError(
            "option price underflowed to zero -- no implied vol can be recovered from this "
            f"quote (forward={forward}, strike={strike}, maturity={maturity})",
            reason="zero_price",
        )

    price_tol = _price_tol(market_price)
    min_vega = _min_vega(forward)

    vol = initial_guess
    for _ in range(_MAX_NEWTON_ITER):
        diff = price(option_type, forward, strike, maturity, rate, vol) - market_price
        vega = greeks(option_type, forward, strike, maturity, rate, vol).vega
        # Only trust the price-difference convergence check once vega
        # confirms the solver is in a regime where a Newton step is
        # actually meaningful. Checking `abs(diff) < price_tol` on its own,
        # before ever looking at vega, is exactly the bug this guards
        # against: near-zero vega means `diff` can be trivially small
        # (both prices near-underflowed) without the solver having taken a
        # single real step -- which would silently return the unmodified
        # initial guess as if it were a real answer.
        if vega >= min_vega and abs(diff) < price_tol:
            return vol
        if vega < min_vega:
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
            f"market_price={market_price} is a valid, arbitrage-free price, but no vol in the "
            f"search range [{lo}, {hi}] reproduces it -- a solver search-range limitation, "
            f"separate from a no-arbitrage violation",
            reason="no_solution_in_range",
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
            f"[{lower_bound:.6f}, {upper_bound:.6f}]",
            reason="no_arbitrage_violation",
        )
