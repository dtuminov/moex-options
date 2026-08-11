"""Turns a ChainSnapshot into a tidy implied-vol surface: one row per liquid,
priceable contract with strike, time-to-expiry, moneyness, and implied vol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from moex_options.black76 import OptionType
from moex_options.chain import ChainSnapshot
from moex_options.implied_vol import ImpliedVolError, solve_implied_vol

_DAYS_PER_YEAR = 365.0  # ACT/365 year-fraction convention


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
    skipped_no_arbitrage: int


def build_surface(snapshot: ChainSnapshot, rate: float) -> SurfaceResult:
    """`rate`: a single flat discount-rate assumption applied to every
    contract — a real simplification (the term structure of Russian
    short-term rates is its own research topic), made explicit here rather
    than silently baked in.
    """
    records: list[dict[str, Any]] = []
    skipped_expired = 0
    skipped_no_arbitrage = 0

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
        except ImpliedVolError:
            skipped_no_arbitrage += 1
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

    return SurfaceResult(
        surface=pd.DataFrame(records),
        skipped_expired=skipped_expired,
        skipped_no_arbitrage=skipped_no_arbitrage,
    )


def select_otm(surface: pd.DataFrame) -> pd.DataFrame:
    """Keeps only out-of-the-money (or exactly ATM) options: puts at
    `moneyness <= 1`, calls at `moneyness >= 1`.

    Standard practice for building a smile, not just a stylistic choice:
    OTM options are the more liquid, more actively quoted side of the
    market, so their prices are the more trustworthy signal of where
    implied vol actually is. At the same strike, an ITM option's price is
    dominated by intrinsic value — a wide bid/offer on a large number
    translates into a much noisier implied vol than the same spread would
    on a small, mostly-time-value OTM price. Using both sides indiscriminately
    is exactly what produces a jagged, zig-zagging plotted "smile" instead of
    a clean one — this filter is the fix for that, not a cosmetic step.
    """
    if surface.empty:
        return surface
    is_otm_put = (surface["option_type"] == OptionType.PUT) & (surface["moneyness"] <= 1.0)
    is_otm_call = (surface["option_type"] == OptionType.CALL) & (surface["moneyness"] >= 1.0)
    return surface[is_otm_put | is_otm_call].reset_index(drop=True)
