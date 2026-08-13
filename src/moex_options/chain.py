"""Live MOEX FORTS option chain + underlying futures snapshot.

Fetches a current cross-sectional snapshot — the full options market's
reference list and live marketdata in one request each, filtered to one
underlying asset — rather than a time series. MOEX's free ISS access gives
live bid/offer only, no history, so this builds today's full strike/
maturity grid instead.

MOEX runs two parallel option series per underlying stock: one struck on
the futures price directly (`SBRF-9.26M160926CA29000`, strike ~29000,
matching the futures price scale) and one struck close to the per-share
price (`SBERP160926CE210`, strike ~210, roughly 1/100th). Mixing them into
one forward-vs-strike comparison would corrupt the surface. `_OPTION_NAME_RE`
matches only the futures-scaled family, the one directly comparable to the
futures marketdata fetched alongside it.

Names that don't match the regex at all, and names that match but have no
corresponding futures forward price, are two distinct failure modes,
counted separately: `skipped_unparsed_names` and `skipped_missing_forward`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import httpx
import pandas as pd

from moex_options.black76 import OptionType

_BASE_URL = "https://iss.moex.com/iss"
# The literal "A" between the C/P type letter and the strike marks the
# American-exercise style of this option series (MOEX FORTS options are
# American-style) -- an unlabeled part of MOEX's own naming convention.
_OPTION_NAME_RE = re.compile(
    r"^(?P<underlying>[A-Z]+-\d+\.\d+)M\d{6}(?P<type>[CP])A(?P<strike>\d+(?:\.\d+)?)$"
)


class MoexChainError(RuntimeError):
    """Raised when the ISS API returns an unexpected payload shape."""


@dataclass(frozen=True, slots=True)
class ChainSnapshot:
    as_of: date
    # columns: secid, underlying_label, option_type, strike, expiry, forward, bid, offer, mid
    rows: pd.DataFrame
    skipped_unparsed_names: int  # name didn't match the futures-scaled family regex at all
    skipped_missing_forward: int  # name matched but no futures forward price was found for it
    skipped_illiquid: int  # contracts with no live bid+offer


class MoexOptionsChainClient:
    def __init__(
        self, base_url: str = _BASE_URL, timeout: float = 15.0, client: httpx.Client | None = None
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def __enter__(self) -> MoexOptionsChainClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def fetch_chain(self, asset_code: str) -> ChainSnapshot:
        option_securities, option_marketdata = self._fetch_market_snapshot("options")
        futures_securities, futures_marketdata = self._fetch_market_snapshot("forts")

        forward_by_label = _build_forward_lookup(futures_securities, futures_marketdata, asset_code)

        rows: list[dict[str, Any]] = []
        skipped_unparsed = 0
        skipped_missing_forward = 0
        skipped_illiquid = 0
        md_by_secid = {r["SECID"]: r for r in option_marketdata}

        for sec in option_securities:
            if sec.get("ASSETCODE") != asset_code:
                continue
            match = _OPTION_NAME_RE.match(sec["SHORTNAME"])
            if match is None:
                skipped_unparsed += 1
                continue

            forward = forward_by_label.get(match.group("underlying"))
            if forward is None:
                skipped_missing_forward += 1
                continue

            md = md_by_secid.get(sec["SECID"])
            bid = float(md["BID"]) if md and md.get("BID") else 0.0
            offer = float(md["OFFER"]) if md and md.get("OFFER") else 0.0
            if bid <= 0 or offer <= 0:
                skipped_illiquid += 1
                continue

            rows.append(
                {
                    "secid": sec["SECID"],
                    "underlying_label": match.group("underlying"),
                    "option_type": OptionType.CALL
                    if match.group("type") == "C"
                    else OptionType.PUT,
                    "strike": float(match.group("strike")),
                    "expiry": datetime.strptime(sec["LASTTRADEDATE"], "%Y-%m-%d").date(),
                    "forward": forward,
                    "bid": bid,
                    "offer": offer,
                    "mid": (bid + offer) / 2.0,
                }
            )

        return ChainSnapshot(
            as_of=date.today(),
            rows=pd.DataFrame(rows),
            skipped_unparsed_names=skipped_unparsed,
            skipped_missing_forward=skipped_missing_forward,
            skipped_illiquid=skipped_illiquid,
        )

    def _fetch_market_snapshot(
        self, market: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        url = f"{self._base_url}/engines/futures/markets/{market}/securities.json"
        params = {"iss.meta": "off", "iss.only": "securities,marketdata"}
        response = self._client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
        try:
            securities = _rows_as_dicts(payload["securities"])
            marketdata = _rows_as_dicts(payload["marketdata"])
        except KeyError as exc:
            raise MoexChainError(f"unexpected ISS payload shape for market={market}") from exc
        return securities, marketdata


def _rows_as_dicts(block: dict[str, Any]) -> list[dict[str, Any]]:
    columns: list[str] = block["columns"]
    data: list[list[Any]] = block["data"]
    return [dict(zip(columns, row, strict=True)) for row in data]


def _build_forward_lookup(
    futures_securities: Iterable[dict[str, Any]],
    futures_marketdata: Iterable[dict[str, Any]],
    asset_code: str,
) -> dict[str, float]:
    md_by_secid = {r["SECID"]: r for r in futures_marketdata}
    forward_by_label: dict[str, float] = {}
    for sec in futures_securities:
        if sec.get("ASSETCODE") != asset_code:
            continue
        md = md_by_secid.get(sec["SECID"])
        if md is None:
            continue
        forward = _best_available_price(md)
        if forward is not None:
            forward_by_label[str(sec["SHORTNAME"])] = forward
    return forward_by_label


def _best_available_price(marketdata_row: dict[str, Any]) -> float | None:
    bid = float(marketdata_row["BID"]) if marketdata_row.get("BID") else 0.0
    offer = float(marketdata_row["OFFER"]) if marketdata_row.get("OFFER") else 0.0
    if bid > 0 and offer > 0:
        return (bid + offer) / 2.0
    last = float(marketdata_row["LAST"]) if marketdata_row.get("LAST") else 0.0
    if last > 0:
        return last
    settle = float(marketdata_row["SETTLEPRICE"]) if marketdata_row.get("SETTLEPRICE") else 0.0
    return settle if settle > 0 else None
