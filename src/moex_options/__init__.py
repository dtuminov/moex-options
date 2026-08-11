"""Black-76 option pricing/Greeks from scratch, an implied-vol solver, and a
real-time MOEX FORTS volatility surface."""

from moex_options.black76 import Greeks, OptionType, greeks, price
from moex_options.chain import ChainSnapshot, MoexChainError, MoexOptionsChainClient
from moex_options.implied_vol import ImpliedVolError, solve_implied_vol
from moex_options.surface import SurfaceResult, build_surface, select_otm

__version__ = "0.1.0"

__all__ = [
    "ChainSnapshot",
    "Greeks",
    "ImpliedVolError",
    "MoexChainError",
    "MoexOptionsChainClient",
    "OptionType",
    "SurfaceResult",
    "build_surface",
    "greeks",
    "price",
    "select_otm",
    "solve_implied_vol",
]
