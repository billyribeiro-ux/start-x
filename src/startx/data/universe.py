"""Load and query the trading universe defined in config/universe.yaml."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_DEFAULT_PATH = Path("config/universe.yaml")


@dataclass(frozen=True)
class SymbolSpec:
    ticker: str       # display ticker, e.g. "SPX"
    fmp: str          # FMP symbol, e.g. "^GSPC"
    type: str         # "stock" | "index_etf"
    name: str

    @property
    def is_stock(self) -> bool:
        return self.type == "stock"


class Universe:
    def __init__(self, data: dict) -> None:
        self.benchmark: str = data.get("benchmark", "^GSPC")
        self._symbols: dict[str, SymbolSpec] = {
            t: SymbolSpec(ticker=t, fmp=v["fmp"], type=v["type"], name=v.get("name", t))
            for t, v in data.get("symbols", {}).items()
        }
        self.context: dict[str, str] = {
            k: v["fmp"] for k, v in data.get("context", {}).items()
        }

    @property
    def tickers(self) -> list[str]:
        return list(self._symbols)

    def spec(self, ticker: str) -> SymbolSpec:
        return self._symbols[ticker]

    def __iter__(self):
        return iter(self._symbols.values())


def load_universe(path: str | Path = _DEFAULT_PATH) -> Universe:
    with open(path, "r", encoding="utf-8") as fh:
        return Universe(yaml.safe_load(fh))
