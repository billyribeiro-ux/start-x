"""qedge execution layer — realistic costs/fills, options bid-ask, and capacity.

The contract is firm: no Sharpe is reported until transaction costs, fees,
realistic slippage, and partial fills are modelled. For options the bid-ask
spread is the dominant cost and is modelled explicitly. Capacity analysis states
the AUM at which the edge dies (market-impact decay), and intraday latency
assumptions are explicit.
"""
from __future__ import annotations
