"""Black-76: option pricing and Greeks for options on a futures/forward
underlying, implemented from scratch (only `scipy.stats.norm` is used, for
the standard normal CDF/PDF — the same primitive any implementation would
reach for, not a pricing library doing the actual work).

**Not vanilla Black-Scholes-Merton, on purpose.** FORTS options are options
on a futures contract. Pricing those with textbook BSM — plugging the
futures price straight in for spot, no adjustment — is a specific, common
mistake: it discounts the strike leg by `exp(-rT)` but leaves the forward
leg undiscounted, i.e. `F*N(d1) - K*exp(-rT)*N(d2)`. Black-76 discounts the
*entire* payoff uniformly: `exp(-rT) * (F*N(d1) - K*N(d2))`. That's what's
actually correct when the underlying is a forward/futures price rather than
a spot price carrying its own cost-of-carry term.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.stats import norm


class OptionType(StrEnum):
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True, slots=True)
class Greeks:
    delta: float
    gamma: float
    vega: float  # per 1.00 (100 vol points) change in vol — divide by 100 for "per 1 vol point"
    theta: float  # per year — divide by 365 for "per day"
    rho: float  # per 1.00 (100 percentage points) change in the discount rate


def _d1_d2(forward: float, strike: float, maturity: float, vol: float) -> tuple[float, float]:
    if forward <= 0 or strike <= 0:
        raise ValueError(f"forward and strike must be positive, got {forward}, {strike}")
    if maturity <= 0:
        raise ValueError(f"maturity must be positive, got {maturity}")
    if vol <= 0:
        raise ValueError(f"vol must be positive, got {vol}")
    sqrt_t = np.sqrt(maturity)
    d1 = (np.log(forward / strike) + 0.5 * vol**2 * maturity) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return float(d1), float(d2)


def price(
    option_type: OptionType, forward: float, strike: float, maturity: float, rate: float, vol: float
) -> float:
    d1, d2 = _d1_d2(forward, strike, maturity, vol)
    discount = np.exp(-rate * maturity)
    if option_type is OptionType.CALL:
        return float(discount * (forward * norm.cdf(d1) - strike * norm.cdf(d2)))
    return float(discount * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1)))


def greeks(
    option_type: OptionType, forward: float, strike: float, maturity: float, rate: float, vol: float
) -> Greeks:
    d1, d2 = _d1_d2(forward, strike, maturity, vol)
    discount = np.exp(-rate * maturity)
    pdf_d1 = float(norm.pdf(d1))
    sqrt_t = np.sqrt(maturity)

    gamma = discount * pdf_d1 / (forward * vol * sqrt_t)
    vega = forward * discount * pdf_d1 * sqrt_t
    theta_common = -forward * pdf_d1 * vol * discount / (2 * sqrt_t)

    if option_type is OptionType.CALL:
        delta = discount * norm.cdf(d1)
        theta = (
            theta_common
            - rate * strike * discount * norm.cdf(d2)
            + rate * forward * discount * norm.cdf(d1)
        )
    else:
        delta = discount * (norm.cdf(d1) - 1.0)
        theta = (
            theta_common
            + rate * strike * discount * norm.cdf(-d2)
            - rate * forward * discount * norm.cdf(-d1)
        )

    option_price = price(option_type, forward, strike, maturity, rate, vol)
    rho = -maturity * option_price

    return Greeks(
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
        rho=float(rho),
    )
