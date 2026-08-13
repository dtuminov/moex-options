"""End-to-end demo: fetch today's full SBRF (Sberbank futures) option chain
from the live MOEX market, solve implied vol for every liquid contract via
the from-scratch Black-76 + Newton/Brent solver, and plot the resulting
smile/surface.

`RISK_FREE_RATE` is a flat rate applied to every maturity — see README
limitations for the term-structure caveat.

Run with: uv run python scripts/run_demo.py
Needs the market open with live quotes (MOEX FORTS trading hours).
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
        f"(skipped {snapshot.skipped_unparsed_names} unparsed names, "
        f"{snapshot.skipped_missing_forward} with no matching futures forward, "
        f"{snapshot.skipped_illiquid} with no live bid+offer)"
    )
    if snapshot.rows.empty:
        raise SystemExit("No liquid contracts right now — try again during MOEX trading hours.")

    result = build_surface(snapshot, rate=RISK_FREE_RATE)
    print(
        f"Solved implied vol for {len(result.surface)} contracts "
        f"(skipped {result.skipped_expired} expired, "
        f"{result.skipped_no_arbitrage} violating no-arbitrage bounds, "
        f"{result.skipped_unsolved} otherwise unsolvable)\n"
    )
    if result.flagged_arbitrage:
        print(f"WARNING: {len(result.flagged_arbitrage)} butterfly-arbitrage flags:")
        for flag in result.flagged_arbitrage:
            print(
                f"  {flag.expiry} {flag.option_type.value}: "
                f"K={flag.strike_low:.0f}/{flag.strike_mid:.0f}/{flag.strike_high:.0f} "
                f"price_mid={flag.price_mid:.2f} > bound={flag.interpolated_bound:.2f} "
                f"(violation={flag.violation:.2f})"
            )
        print()

    # OTM filter rationale: see select_otm docstring in surface.py.
    # Plots are grouped by expiry, not underlying_label -- see SurfaceResult
    # docstring in surface.py.
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

    # Strikes involved in a butterfly-arbitrage flag, for marking on the plot.
    flagged_strikes = {
        (flag.expiry, flag.option_type, strike)
        for flag in result.flagged_arbitrage
        for strike in (flag.strike_low, flag.strike_mid, flag.strike_high)
    }

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    flagged_label_used = False
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
        is_flagged = ordered.apply(
            lambda row: (row["expiry"], row["option_type"], row["strike"]) in flagged_strikes,
            axis=1,
        )
        if is_flagged.any():
            flagged_points = ordered[is_flagged]
            ax.scatter(
                flagged_points["moneyness"],
                flagged_points["implied_vol"] * 100,
                marker="x",
                color="red",
                s=60,
                zorder=5,
                label=None if flagged_label_used else "butterfly-arbitrage flag",
            )
            flagged_label_used = True
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
