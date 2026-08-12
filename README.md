# moex-options

Black-76 option pricing and Greeks, a Newton/Brent implied-vol solver, and a
volatility surface built from the live MOEX FORTS options market.

## Contents

- `src/moex_options/black76.py` — Black-76 pricing and Greeks for an option
  on a futures/forward underlying.
- `src/moex_options/implied_vol.py` — inverts Black-76 for implied vol via
  Newton-Raphson, falling back to Brent's method.
- `src/moex_options/chain.py` — fetches the current SBRF option chain and
  underlying futures quotes from MOEX ISS.
- `src/moex_options/surface.py` — builds an implied-vol surface from a chain
  snapshot: OTM filter, plus a butterfly-arbitrage diagnostic.
- `scripts/run_demo.py` — end-to-end run against the live market; writes
  `reports/assets/vol_smile.png`.

## Methodology

- **Black-76, not Black-Scholes-Merton.** FORTS options are options on a
  futures contract. Plugging the futures price into textbook BSM discounts
  the strike leg but not the forward leg (`F*N(d1) - K*exp(-rT)*N(d2)`);
  Black-76 discounts the whole payoff (`exp(-rT)*(F*N(d1) - K*N(d2))`).
  Details in the `black76.py` module docstring; `test_black76.py` checks
  put-call parity and the delta-difference-equals-discount-factor identity
  exactly.
- **Live snapshot, not a historical series.** MOEX's free ISS access gives
  live bid/offer, no history, so this builds today's full strike/maturity
  grid instead of a time series. Details in `chain.py`.
- **Two option-name scales per underlying.** MOEX runs a futures-scaled
  series (`SBRF-9.26M160926CA29000`, strike ~29000) and a share-scaled
  series (`SBERP160926CE210`, strike ~210) per stock. Only the futures-
  scaled family parses. Names that fail to parse and names that parse but
  have no matching futures forward are counted separately
  (`skipped_unparsed_names`, `skipped_missing_forward`).
- **OTM-only smile.** An ITM option's price is dominated by intrinsic
  value, so a given bid/offer spread implies a noisier vol than the same
  spread on an OTM price. `select_otm` keeps only the OTM (or ATM) side per
  strike.
- **Butterfly-arbitrage check.** For each expiry and option type,
  `check_butterfly_arbitrage` walks adjacent strike triples K1<K2<K3 and
  flags cases where `price(K2)` exceeds the strike-weighted interpolation
  of `price(K1)` and `price(K3)` — a convexity violation, i.e. a negative
  butterfly spread. Runs on raw quoted mid-prices, not a fitted curve;
  `SurfaceResult.flagged_arbitrage` carries the flags.

## Results (SBRF, 2026-08-12, MOEX open)

| Stage | Count |
|---|---|
| SBRF option names seen | 2,360 |
| Skipped — unparsed name | 1,744 |
| Skipped — no matching futures forward | 0 |
| Skipped — no live bid+offer | 538 |
| Liquid contracts | 78 |
| Skipped — expired | 3 |
| Skipped — violates single-contract no-arbitrage bound | 0 |
| Implied vol solved | 75 |
| Flagged — butterfly arbitrage | 20 |
| Kept after OTM filter | 57 |
| Distinct expiries | 4 |

![SBRF implied vol smile by expiry](reports/assets/vol_smile.png)

Red X markers are strikes involved in a butterfly-arbitrage flag: 20 of 75
solved contracts. 14 of the 20 flags are in the 2026-09-16 series (31
contracts, the deepest liquidity of the four) — dense, closely-spaced
strikes with real bid/offer noise on raw mid-prices trip the convexity
check more easily than sparser series do. The check runs on unsmoothed
quotes; a fitted surface would resolve some of what it flags.

```bash
uv run python scripts/run_demo.py
```

Fetches live SBRF quotes, prints the near-the-money strip of every expiry
and any butterfly-arbitrage flags, and writes `reports/assets/vol_smile.png`.

## Installation & usage

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python scripts/run_demo.py   # needs MOEX FORTS trading hours for live quotes
uv run pytest                        # HTTP mocked, no network needed
uv run ruff check . && uv run mypy src tests
```

CI (`.github/workflows/ci.yml`) runs lint, type check, and tests on every
push and PR.

## Roadmap

- Fetch RUONIA (or another short-rate series) instead of the flat
  `RISK_FREE_RATE` assumption.
- A parametric smile fit (SVI or similar) over the OTM points.
- Feed the fitted surface into `moex-cvar-portfolio` as an options overlay,
  or price a variance-swap replication from it.

## Limitations

- Flat discount rate (`RISK_FREE_RATE = 0.15`) across all maturities; no
  term structure.
- No surface interpolation or smile fit — each point is an independently
  solved implied vol; the surface is a raw scatter.
- The butterfly-arbitrage check runs on raw mid-prices from live bid/offer
  quotes, not a fitted curve, so flags mix genuine mispricing with
  bid/offer noise; it does not distinguish the two.
- Live snapshot only — no historical vol surface, a data-source limitation
  of MOEX's free ISS access, not an implementation choice.
- Standalone project, not wired into any backtest engine.
