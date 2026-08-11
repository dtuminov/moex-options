"""End-to-end demo on the REAL, LIVE MOEX options market: fetch today's
full SBRF (Sberbank futures) option chain, solve implied vol for every
liquid contract via the from-scratch Black-76 + Newton/bisection solver,
and plot the resulting smile/surface.

**`RISK_FREE_RATE` is a flat assumption, not fetched.** The term structure
of Russian short-term rates is its own research topic; a single flat rate
across all maturities is a simplification, made explicit here rather than
silently baked into the surface.

Run with: uv run python scripts/run_demo.py
(Needs the market to be open with live quotes; MOEX FORTS trading hours.)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from moex_options.chain import MoexOptionsChainClient
from moex_options.surface import build_surface, select_otm

ASSET_CODE = "SBRF"
RISK_FREE_RATE = 0.15  # flat assumption — see module docstring
ASSETS_DIR = Path(__file__).resolve().parent.parent / "reports" / "assets"


def main() -> None:
    with MoexOptionsChainClient() as client:
        snapshot = client.fetch_chain(ASSET_CODE)

    print(
        f"{ASSET_CODE} chain as of {snapshot.as_of}: {len(snapshot.rows)} liquid contracts "
        f"(skipped {snapshot.skipped_unparsed_names} other-scale/other-asset names, "
        f"{snapshot.skipped_illiquid} with no live bid+offer)"
    )
    if snapshot.rows.empty:
        raise SystemExit("No liquid contracts right now — try again during MOEX trading hours.")

    result = build_surface(snapshot, rate=RISK_FREE_RATE)
    print(
        f"Solved implied vol for {len(result.surface)} contracts "
        f"(skipped {result.skipped_expired} expired, "
        f"{result.skipped_no_arbitrage} violating no-arbitrage bounds)\n"
    )

    # Group by actual expiry, not underlying_label: a monthly series and
    # several weeklies commonly share the same underlying_label because
    # they're struck against the same futures contract, but they are
    # different maturities and must not be plotted as one smile.
    # OTM-only: at the same strike, ITM and OTM quotes can imply visibly
    # different vol once real bid/offer noise is in the picture (mid-price
    # of a large, intrinsic-value-dominated ITM quote is a noisier signal
    # than the same spread on a small, mostly-time-value OTM one) — using
    # both sides indiscriminately is what makes a smile plot zig-zag.
    surface = select_otm(result.surface)
    print(f"Using {len(surface)}/{len(result.surface)} contracts after keeping OTM-only\n")
    for expiry, group in surface.groupby("expiry"):
        maturity = group["maturity_years"].iloc[0]
        underlying = group["underlying_label"].iloc[0]
        print(f"{underlying} expiring {expiry} (T={maturity:.3f}y, {len(group)} contracts):")
        near_money = group.iloc[(group["moneyness"] - 1.0).abs().argsort()[:5]]
        for _, row in near_money.sort_values("moneyness").iterrows():
            print(
                f"  strike={row['strike']:>10,.0f}  moneyness={row['moneyness']:.3f}  "
                f"{row['option_type']:<4}  IV={row['implied_vol']:.1%}"
            )
        print()

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for expiry, group in surface.groupby("expiry"):
        maturity = group["maturity_years"].iloc[0]
        ordered = group.sort_values("moneyness")
        ax.plot(
            ordered["moneyness"],
            ordered["implied_vol"] * 100,
            "o-",
            label=f"{expiry} (T={maturity:.2f}y)",
            alpha=0.7,
        )
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, label="ATM (moneyness=1)")
    ax.set_xlabel("Moneyness (strike / forward)")
    ax.set_ylabel("Implied volatility, %")
    ax.set_title(f"{ASSET_CODE}: implied vol smile by expiry, {snapshot.as_of}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png_path = ASSETS_DIR / "vol_smile.png"
    fig.savefig(png_path, dpi=150)
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
