"""Turns a ChainSnapshot into a tidy implied-vol surface: one row per liquid,
priceable contract with strike, time-to-expiry, moneyness, and implied vol.

Also runs a static no-arbitrage check across strikes within each expiry —
see `check_butterfly_arbitrage`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from moex_options.black76 import OptionType
from moex_options.chain import ChainSnapshot
from moex_options.implied_vol import ImpliedVolError, solve_implied_vol

_DAYS_PER_YEAR = 365.0  # ACT/365 year-fraction convention
# Relative price tolerance (fraction of the interpolated bound), to absorb
# float rounding without being swamped by it. An absolute 1e-6 has no
# connection to the actual scale of MOEX premiums (hundreds to thousands of
# rubles), so it's scaled to the bound itself, floored at 1.0 so it stays
# sane for near-zero bounds.
_BUTTERFLY_TOLERANCE_REL = 1e-6


@dataclass(frozen=True, slots=True)
class ButterflyFlag:
    """One adjacent-strike triple (K1 < K2 < K3), same expiry and option
    type, where the observed mid-price at K2 exceeds the strike-weighted
    interpolation of the mid-prices at K1 and K3 — a violation of convexity
    of price in strike, i.e. a negative butterfly spread. See
    `check_butterfly_arbitrage`.
    """

    expiry: date
    option_type: OptionType
    strike_low: float
    strike_mid: float
    strike_high: float
    price_low: float
    price_mid: float
    price_high: float
    interpolated_bound: float

    @property
    def violation(self) -> float:
        """How much price_mid exceeds the no-arbitrage bound, in price units."""
        return self.price_mid - self.interpolated_bound


@dataclass(frozen=True, slots=True)
class SurfaceResult:
    # columns: secid, underlying_label, expiry, option_type, strike, forward,
    #          moneyness, maturity_years, mid_price, implied_vol
    #
    # Group by `expiry` (not `underlying_label`) for a clean per-maturity
    # smile: several distinct expiries — a monthly series and multiple
    # weeklies — commonly share the same `underlying_label`, because they're
    # all struck against the same underlying futures contract.
    surface: pd.DataFrame
    skipped_expired: int
    skipped_no_arbitrage: int  # ImpliedVolError, reason="no_arbitrage_violation"
    skipped_unsolved: int  # ImpliedVolError, any other reason (zero-price underflow,
    # no solution in the solver's search range) -- a contract that simply couldn't be
    # given an implied vol, separate from an arbitrage violation. Kept apart from
    # skipped_no_arbitrage since lumping them together would repeat the exact
    # misleading-naming problem `ImpliedVolError.reason` exists to avoid.
    flagged_arbitrage: list[ButterflyFlag]


def build_surface(snapshot: ChainSnapshot, rate: float) -> SurfaceResult:
    """`rate`: a flat discount rate applied to every contract. The term
    structure of Russian short-term rates isn't modeled — see README
    limitations.
    """
    records: list[dict[str, Any]] = []
    skipped_expired = 0
    skipped_no_arbitrage = 0
    skipped_unsolved = 0

    for row in snapshot.rows.to_dict("records"):
        expiry = row["expiry"]
        strike = float(row["strike"])
        forward = float(row["forward"])
        maturity = (expiry - snapshot.as_of).days / _DAYS_PER_YEAR
        if maturity <= 0:
            skipped_expired += 1
            continue
        try:
            implied_vol = solve_implied_vol(
                row["option_type"], float(row["mid"]), forward, strike, maturity, rate
            )
        except ImpliedVolError as exc:
            if exc.reason == "no_arbitrage_violation":
                skipped_no_arbitrage += 1
            else:
                skipped_unsolved += 1
            continue

        records.append(
            {
                "secid": row["secid"],
                "underlying_label": row["underlying_label"],
                "expiry": expiry,
                "option_type": row["option_type"],
                "strike": strike,
                "forward": forward,
                "moneyness": strike / forward,
                "maturity_years": maturity,
                "mid_price": float(row["mid"]),
                "implied_vol": implied_vol,
            }
        )

    surface_df = pd.DataFrame(records)
    return SurfaceResult(
        surface=surface_df,
        skipped_expired=skipped_expired,
        skipped_no_arbitrage=skipped_no_arbitrage,
        skipped_unsolved=skipped_unsolved,
        flagged_arbitrage=check_butterfly_arbitrage(surface_df),
    )


def check_butterfly_arbitrage(surface: pd.DataFrame) -> list[ButterflyFlag]:
    """Static no-arbitrage check: for every (expiry, option_type) group,
    walk adjacent strike triples K1 < K2 < K3 and flag any where

        price(K2) > w1 * price(K1) + w3 * price(K3)

    with w1, w3 the linear-interpolation weights of K2 between K1 and K3
    (w1 = (K3-K2)/(K3-K1), w3 = (K2-K1)/(K3-K1); reduces to the textbook
    (price(K1)+price(K3))/2 bound when strikes are evenly spaced). A long
    K1 + long K3 butterfly, weighted to match K2, must be worth at least as
    much as an equivalent position in K2; a violation means the three
    quoted prices imply a butterfly that can be sold for a riskless profit.

    Checked per option type, keeping calls and puts separate: price is
    convex and monotonic in strike for a single option type over its full
    range, but an OTM put and an OTM call either side of the forward don't
    form a convex sequence without a put-call-parity adjustment first. This
    is a v1 diagnostic on raw quoted prices — no interpolation or smile fit
    is applied beyond the three-point check itself.
    """
    flags: list[ButterflyFlag] = []
    if surface.empty:
        return flags

    for _, group in surface.groupby(["expiry", "option_type"]):
        ordered = group.sort_values("strike")
        expiry: date = ordered["expiry"].iloc[0]
        option_type: OptionType = ordered["option_type"].iloc[0]
        strikes = ordered["strike"].to_numpy()
        prices = ordered["mid_price"].to_numpy()
        for i in range(len(strikes) - 2):
            k1, k2, k3 = strikes[i], strikes[i + 1], strikes[i + 2]
            if k3 == k1:
                continue  # duplicate strikes, nothing to interpolate
            p1, p2, p3 = prices[i], prices[i + 1], prices[i + 2]
            w1 = (k3 - k2) / (k3 - k1)
            w3 = (k2 - k1) / (k3 - k1)
            bound = w1 * p1 + w3 * p3
            tolerance = _BUTTERFLY_TOLERANCE_REL * max(abs(bound), 1.0)
            if p2 > bound + tolerance:
                flags.append(
                    ButterflyFlag(
                        expiry=expiry,
                        option_type=option_type,
                        strike_low=float(k1),
                        strike_mid=float(k2),
                        strike_high=float(k3),
                        price_low=float(p1),
                        price_mid=float(p2),
                        price_high=float(p3),
                        interpolated_bound=float(bound),
                    )
                )
    return flags


def select_otm(surface: pd.DataFrame) -> pd.DataFrame:
    """Keeps only out-of-the-money (or exactly ATM) options: puts at
    `moneyness <= 1`, calls at `moneyness >= 1`.

    OTM options are the more liquid side of the market at a given strike.
    An ITM option's price is dominated by intrinsic value, so a given
    bid/offer spread implies a noisier vol than the same spread on a
    smaller, mostly-time-value OTM price. Mixing both sides produces a
    jagged smile.
    """
    if surface.empty:
        return surface
    is_otm_put = (surface["option_type"] == OptionType.PUT) & (surface["moneyness"] <= 1.0)
    is_otm_call = (surface["option_type"] == OptionType.CALL) & (surface["moneyness"] >= 1.0)
    return surface[is_otm_put | is_otm_call].reset_index(drop=True)
