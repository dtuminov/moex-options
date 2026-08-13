"""Tests for build_surface."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from moex_options.black76 import OptionType, price
from moex_options.chain import ChainSnapshot
from moex_options.surface import ButterflyFlag, build_surface, check_butterfly_arbitrage, select_otm

_TODAY = date(2026, 8, 11)


def _snapshot(rows: list[dict[str, object]]) -> ChainSnapshot:
    return ChainSnapshot(
        as_of=_TODAY,
        rows=pd.DataFrame(rows),
        skipped_unparsed_names=0,
        skipped_missing_forward=0,
        skipped_illiquid=0,
    )


def test_recovers_implied_vol_for_a_valid_liquid_contract() -> None:
    forward, strike, rate, true_vol = 29000.0, 30000.0, 0.15, 0.35
    maturity_days = 90
    maturity_years = maturity_days / 365.0
    mid_price = price(OptionType.CALL, forward, strike, maturity_years, rate, true_vol)

    snapshot = _snapshot(
        [
            {
                "secid": "SR30000BI6",
                "underlying_label": "SBRF-9.26",
                "option_type": OptionType.CALL,
                "strike": strike,
                "expiry": _TODAY + timedelta(days=maturity_days),
                "forward": forward,
                "bid": mid_price - 1,
                "offer": mid_price + 1,
                "mid": mid_price,
            }
        ]
    )

    result = build_surface(snapshot, rate=rate)

    assert len(result.surface) == 1
    row = result.surface.iloc[0]
    assert row["implied_vol"] == pytest.approx(true_vol, abs=1e-4)
    assert row["moneyness"] == pytest.approx(strike / forward)
    assert result.skipped_expired == 0
    assert result.skipped_no_arbitrage == 0
    assert result.flagged_arbitrage == []


def test_expired_contract_is_skipped_and_counted() -> None:
    snapshot = _snapshot(
        [
            {
                "secid": "SR30000BI6",
                "underlying_label": "SBRF-9.26",
                "option_type": OptionType.CALL,
                "strike": 30000.0,
                "expiry": _TODAY - timedelta(days=1),  # already expired
                "forward": 29000.0,
                "bid": 100.0,
                "offer": 110.0,
                "mid": 105.0,
            }
        ]
    )

    result = build_surface(snapshot, rate=0.15)

    assert result.surface.empty
    assert result.skipped_expired == 1
    assert result.skipped_no_arbitrage == 0


def test_no_arbitrage_violating_price_is_skipped_and_counted_not_raised() -> None:
    snapshot = _snapshot(
        [
            {
                "secid": "SR30000BI6",
                "underlying_label": "SBRF-9.26",
                "option_type": OptionType.CALL,
                "strike": 30000.0,
                "expiry": _TODAY + timedelta(days=90),
                "forward": 29000.0,
                "bid": 1_000_000.0,  # absurd, above the discount*forward upper bound
                "offer": 1_000_001.0,
                "mid": 1_000_000.5,
            }
        ]
    )

    result = build_surface(snapshot, rate=0.15)

    assert result.surface.empty
    assert result.skipped_no_arbitrage == 1
    assert result.skipped_unsolved == 0


def test_unsolvable_but_arbitrage_free_price_is_counted_separately_from_no_arbitrage() -> None:
    # A market_price of exactly 0.0 is within the no-arbitrage bounds for a
    # deep-OTM contract (lower bound is 0 for an OTM call) -- it's not an
    # arbitrage violation, just a quote solve_implied_vol can't recover a
    # vol from. This must land in skipped_unsolved, not skipped_no_arbitrage
    # -- conflating the two would repeat the exact misleading-naming problem
    # ImpliedVolError.reason exists to avoid (see surface.py SurfaceResult).
    snapshot = _snapshot(
        [
            {
                "secid": "SR100000BI6",
                "underlying_label": "SBRF-9.26",
                "option_type": OptionType.CALL,
                "strike": 100000.0,  # deep OTM: forward is 29000
                "expiry": _TODAY + timedelta(days=1),  # short-dated
                "forward": 29000.0,
                "bid": 0.0,
                "offer": 0.0,
                "mid": 0.0,
            }
        ]
    )

    result = build_surface(snapshot, rate=0.15)

    assert result.surface.empty
    assert result.skipped_no_arbitrage == 0
    assert result.skipped_unsolved == 1


def test_empty_chain_produces_an_empty_surface() -> None:
    result = build_surface(_snapshot([]), rate=0.15)

    assert result.surface.empty
    assert result.skipped_expired == 0
    assert result.skipped_no_arbitrage == 0
    assert result.skipped_unsolved == 0
    assert result.flagged_arbitrage == []


def test_select_otm_keeps_puts_below_forward_and_calls_above() -> None:
    surface = pd.DataFrame(
        [
            {"option_type": OptionType.PUT, "moneyness": 0.9},  # OTM put -> keep
            {"option_type": OptionType.PUT, "moneyness": 1.1},  # ITM put -> drop
            {"option_type": OptionType.CALL, "moneyness": 1.1},  # OTM call -> keep
            {"option_type": OptionType.CALL, "moneyness": 0.9},  # ITM call -> drop
        ]
    )

    result = select_otm(surface)

    assert len(result) == 2
    assert set(result["moneyness"]) == {0.9, 1.1}


def test_select_otm_keeps_both_sides_exactly_at_the_money() -> None:
    surface = pd.DataFrame(
        [
            {"option_type": OptionType.PUT, "moneyness": 1.0},
            {"option_type": OptionType.CALL, "moneyness": 1.0},
        ]
    )

    result = select_otm(surface)

    assert len(result) == 2


def test_select_otm_on_empty_surface_returns_empty() -> None:
    empty = pd.DataFrame(columns=["option_type", "moneyness"])

    result = select_otm(empty)

    assert result.empty


def _butterfly_surface(prices: list[float], strikes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "expiry": [_TODAY] * len(strikes),
            "option_type": [OptionType.CALL] * len(strikes),
            "strike": strikes,
            "mid_price": prices,
        }
    )


def test_check_butterfly_arbitrage_flags_a_convexity_violation() -> None:
    # Evenly spaced strikes; middle price is above the average of the wings
    # -> a butterfly built from these three quotes sells for a riskless
    # profit.
    surface = _butterfly_surface(prices=[10.0, 9.0, 6.0], strikes=[95.0, 100.0, 105.0])

    flags = check_butterfly_arbitrage(surface)

    assert len(flags) == 1
    flag = flags[0]
    assert isinstance(flag, ButterflyFlag)
    assert (flag.strike_low, flag.strike_mid, flag.strike_high) == (95.0, 100.0, 105.0)
    assert flag.interpolated_bound == pytest.approx(8.0)  # (10+6)/2, evenly spaced
    assert flag.violation == pytest.approx(1.0)


def test_check_butterfly_arbitrage_does_not_flag_a_convex_price_curve() -> None:
    # Monotonically decreasing, convex prices in strike -- the normal shape
    # for a single option type's price curve; no arbitrage here.
    surface = _butterfly_surface(prices=[10.0, 6.0, 4.0], strikes=[95.0, 100.0, 105.0])

    flags = check_butterfly_arbitrage(surface)

    assert flags == []


def test_check_butterfly_arbitrage_uses_strike_weighted_interpolation_for_uneven_spacing() -> None:
    # K2 is closer to K1 than K3, so it should be weighted more heavily
    # toward price_low. Bound = 0.75*p1 + 0.25*p3 = 0.75*10 + 0.25*2 = 8.0.
    surface = _butterfly_surface(prices=[10.0, 7.5, 2.0], strikes=[90.0, 95.0, 110.0])

    flags = check_butterfly_arbitrage(surface)

    assert flags == []  # 7.5 <= 8.0, no violation

    # Bump price_mid just over the bound to confirm the weighting is what's
    # doing the work (an even-spacing bound of (10+2)/2=6 would already
    # flag 7.5; this confirms it's the 8.0 uneven-spacing bound in effect).
    surface_flagged = _butterfly_surface(prices=[10.0, 8.5, 2.0], strikes=[90.0, 95.0, 110.0])

    flags_after_bump = check_butterfly_arbitrage(surface_flagged)

    assert len(flags_after_bump) == 1
    assert flags_after_bump[0].interpolated_bound == pytest.approx(8.0)


def test_check_butterfly_arbitrage_on_empty_surface_returns_empty() -> None:
    empty = pd.DataFrame(columns=["expiry", "option_type", "strike", "mid_price"])

    assert check_butterfly_arbitrage(empty) == []
