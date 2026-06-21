# Dashboard pages — why the numbering starts at 4

Streamlit auto-discovers the files in this directory and orders the sidebar by the
leading number in each filename. The numbers are **sidebar sort keys only** — no Python
code imports these files (the leading-digit names aren't even valid module identifiers),
so the gap is cosmetic and safe.

## Current pages
- `4_Reversal_Lab.py` — Reversal Lab
- `5_Forward_Paper.py` — Forward / paper-trading view
- `6_Evidence.py` — Evidence Miner (cross-year up-vs-down conditions)
- `7_Self_Learning.py` — Loss-autopsy + point-in-time meta gate (SYNTHETIC-DATA DEMO)

## Where 1-3 went
There are no pages 1-3. Per git history this directory has only ever contained pages
4-7 — slots 1-3 were never built (they were reserved during planning and the first
shipped pages simply started at 4). The home/landing view is `../app.py` (the "Move
Explorer"), which is the un-numbered root and renders first in the sidebar.

Renumbering to a contiguous 1..N was deliberately avoided: it would only reshuffle
sidebar labels while risking churn against bookmarks/screenshots, and buys no
functional benefit. If you do renumber later, a plain `git mv` is safe — nothing
imports these files.
