"""Tests for MoexOptionsChainClient. All HTTP is mocked via respx — no network access."""

from __future__ import annotations

from datetime import date

import httpx
import pytest
import respx

from moex_options.black76 import OptionType
from moex_options.chain import MoexChainError, MoexOptionsChainClient

_OPTIONS_URL = "https://iss.moex.com/iss/engines/futures/markets/options/securities.json"
_FORTS_URL = "https://iss.moex.com/iss/engines/futures/markets/forts/securities.json"


def _payload(
    sec_columns: list[str],
    sec_rows: list[list[object]],
    md_columns: list[str],
    md_rows: list[list[object]],
) -> dict[str, object]:
    return {
        "securities": {"columns": sec_columns, "data": sec_rows},
        "marketdata": {"columns": md_columns, "data": md_rows},
    }


_SEC_COLUMNS = ["SECID", "SHORTNAME", "ASSETCODE", "LASTTRADEDATE"]
_OPTION_MD_COLUMNS = ["SECID", "BID", "OFFER", "LAST", "SETTLEPRICE", "NUMTRADES"]
_FUT_MD_COLUMNS = ["SECID", "BID", "OFFER", "LAST", "SETTLEPRICE"]


def _mock_options_and_futures(
    option_sec_rows: list[list[object]],
    option_md_rows: list[list[object]],
    futures_sec_rows: list[list[object]],
    futures_md_rows: list[list[object]],
) -> None:
    respx.get(_OPTIONS_URL).mock(
        return_value=httpx.Response(
            200, json=_payload(_SEC_COLUMNS, option_sec_rows, _OPTION_MD_COLUMNS, option_md_rows)
        )
    )
    respx.get(_FORTS_URL).mock(
        return_value=httpx.Response(
            200, json=_payload(_SEC_COLUMNS, futures_sec_rows, _FUT_MD_COLUMNS, futures_md_rows)
        )
    )


@respx.mock
def test_parses_a_liquid_futures_scaled_option_into_a_row() -> None:
    _mock_options_and_futures(
        option_sec_rows=[["SR29000BI6", "SBRF-9.26M160926CA29000", "SBRF", "2026-09-16"]],
        option_md_rows=[["SR29000BI6", 872.0, 992.0, 850.0, 0.0, 6]],
        futures_sec_rows=[["SRU6", "SBRF-9.26", "SBRF", "2026-09-17"]],
        futures_md_rows=[["SRU6", 29063.0, 29066.0, 29064.0, 29064.0]],
    )

    with MoexOptionsChainClient() as client:
        snapshot = client.fetch_chain("SBRF")

    assert len(snapshot.rows) == 1
    row = snapshot.rows.iloc[0]
    assert row["secid"] == "SR29000BI6"
    assert row["option_type"] == OptionType.CALL
    assert row["strike"] == pytest.approx(29000.0)
    assert row["expiry"] == date(2026, 9, 16)
    assert row["forward"] == pytest.approx(29064.5)  # mid of futures bid/offer
    assert row["mid"] == pytest.approx((872.0 + 992.0) / 2.0)
    assert snapshot.skipped_unparsed_names == 0
    assert snapshot.skipped_missing_forward == 0
    assert snapshot.skipped_illiquid == 0


@respx.mock
def test_skips_the_share_scaled_sibling_series() -> None:
    _mock_options_and_futures(
        option_sec_rows=[
            ["SR29000BI6", "SBRF-9.26M160926CA29000", "SBRF", "2026-09-16"],
            [
                "AF22.5CH6",
                "AFLTP190826CE22.5",
                "AFLT",
                "2026-08-19",
            ],  # different scale, different asset
        ],
        option_md_rows=[
            ["SR29000BI6", 872.0, 992.0, 850.0, 0.0, 6],
            ["AF22.5CH6", 1.0, 2.0, 1.5, 0.0, 1],
        ],
        futures_sec_rows=[["SRU6", "SBRF-9.26", "SBRF", "2026-09-17"]],
        futures_md_rows=[["SRU6", 29063.0, 29066.0, 29064.0, 29064.0]],
    )

    with MoexOptionsChainClient() as client:
        snapshot = client.fetch_chain("SBRF")

    assert (
        len(snapshot.rows) == 1
    )  # the AFLT row is a different asset entirely, filtered before parsing
    assert snapshot.skipped_unparsed_names == 0


@respx.mock
def test_skips_contracts_without_a_live_bid_and_offer() -> None:
    _mock_options_and_futures(
        option_sec_rows=[
            ["SR29000BI6", "SBRF-9.26M160926CA29000", "SBRF", "2026-09-16"],
            ["SR30000BI6", "SBRF-9.26M160926CA30000", "SBRF", "2026-09-16"],
        ],
        option_md_rows=[
            ["SR29000BI6", 872.0, 992.0, 850.0, 0.0, 6],
            ["SR30000BI6", 0.0, 0.0, 0.0, 700.0, 0],  # no live bid/offer, settlement only
        ],
        futures_sec_rows=[["SRU6", "SBRF-9.26", "SBRF", "2026-09-17"]],
        futures_md_rows=[["SRU6", 29063.0, 29066.0, 29064.0, 29064.0]],
    )

    with MoexOptionsChainClient() as client:
        snapshot = client.fetch_chain("SBRF")

    assert len(snapshot.rows) == 1
    assert snapshot.skipped_illiquid == 1


@respx.mock
def test_falls_back_to_settlement_price_for_the_forward_when_futures_has_no_live_quote() -> None:
    _mock_options_and_futures(
        option_sec_rows=[["SR29000BI6", "SBRF-9.26M160926CA29000", "SBRF", "2026-09-16"]],
        option_md_rows=[["SR29000BI6", 872.0, 992.0, 850.0, 0.0, 6]],
        futures_sec_rows=[["SRU6", "SBRF-9.26", "SBRF", "2026-09-17"]],
        futures_md_rows=[["SRU6", 0.0, 0.0, 0.0, 29500.0]],  # no bid/offer/last, only settlement
    )

    with MoexOptionsChainClient() as client:
        snapshot = client.fetch_chain("SBRF")

    assert snapshot.rows.iloc[0]["forward"] == pytest.approx(29500.0)


@respx.mock
def test_option_with_no_matching_futures_label_is_skipped() -> None:
    # Name matches the futures-scaled family regex fine, but no futures
    # forward price is known for that underlying label -> skipped_missing_forward,
    # not skipped_unparsed_names.
    _mock_options_and_futures(
        option_sec_rows=[["SR29000BI6", "SBRF-12.26M161216CA29000", "SBRF", "2026-12-16"]],
        option_md_rows=[["SR29000BI6", 872.0, 992.0, 850.0, 0.0, 6]],
        futures_sec_rows=[["SRU6", "SBRF-9.26", "SBRF", "2026-09-17"]],  # only Sept future known
        futures_md_rows=[["SRU6", 29063.0, 29066.0, 29064.0, 29064.0]],
    )

    with MoexOptionsChainClient() as client:
        snapshot = client.fetch_chain("SBRF")

    assert snapshot.rows.empty
    assert snapshot.skipped_unparsed_names == 0
    assert snapshot.skipped_missing_forward == 1


@respx.mock
def test_unparseable_name_is_skipped_and_counted_separately_from_missing_forward() -> None:
    # Regex requires option type C or P; "X" doesn't match at all -> this is
    # the distinct skipped_unparsed_names failure mode.
    _mock_options_and_futures(
        option_sec_rows=[["SR29000BI6", "SBRF-9.26M160926XA29000", "SBRF", "2026-09-16"]],
        option_md_rows=[["SR29000BI6", 872.0, 992.0, 850.0, 0.0, 6]],
        futures_sec_rows=[["SRU6", "SBRF-9.26", "SBRF", "2026-09-17"]],
        futures_md_rows=[["SRU6", 29063.0, 29066.0, 29064.0, 29064.0]],
    )

    with MoexOptionsChainClient() as client:
        snapshot = client.fetch_chain("SBRF")

    assert snapshot.rows.empty
    assert snapshot.skipped_unparsed_names == 1
    assert snapshot.skipped_missing_forward == 0


@respx.mock
def test_unexpected_payload_shape_raises_moex_chain_error() -> None:
    respx.get(_OPTIONS_URL).mock(return_value=httpx.Response(200, json={"unexpected": {}}))
    respx.get(_FORTS_URL).mock(
        return_value=httpx.Response(200, json=_payload(_SEC_COLUMNS, [], _FUT_MD_COLUMNS, []))
    )

    with MoexOptionsChainClient() as client, pytest.raises(MoexChainError):
        client.fetch_chain("SBRF")
