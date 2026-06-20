"""Strategy runner: dataset → walk-forward → backtest → validation scoreboard."""
from .pipeline import StrategyResult, run_per_symbol, run_pooled

__all__ = ["StrategyResult", "run_pooled", "run_per_symbol"]
