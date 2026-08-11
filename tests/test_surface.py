"""Tests for build_surface."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from moex_options.black76 import OptionType, price
from moex_options.chain import ChainSnapshot
from moex_options.surface import build_surface, select_otm

_TODAY = date(2026, 8, 11)


def _snapshot(rows: list[dict[str, object]]) -> ChainSnapshot:
    return ChainSnapshot(
        as_of=_TODAY, rows=pd.DataFrame(rows), skipped_unparsed_names=0, skipped_illiquid=0
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


def test_empty_chain_produces_an_empty_surface() -> None:
    result = build_surface(_snapshot([]), rate=0.15)

    assert result.surface.empty
    assert result.skipped_expired == 0
    assert result.skipped_no_arbitrage == 0


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
