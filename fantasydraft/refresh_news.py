#!/usr/bin/env python
"""Refresh the automated news cache (Sleeper buzz + ESPN reports).

Fast, network-only, no model recompute. Run it, then rebuild the presentation:

    .venv/bin/python refresh_news.py
    .venv/bin/python run_pipeline.py --stage spreadsheet

Populates the Notes column for every draftable player with what the crowd is
moving on and the latest headlines. Curated notes in config/news.yaml and auto
injury flags always take precedence - this only fills the empty cells. Needs
data/processed/draft_board.parquet to exist (run the pipeline at least once).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import news_auto  # noqa: E402


def main() -> int:
    print("=== Refresh auto-news: Sleeper buzz + ESPN reports ===")
    df = news_auto.refresh()
    up = (df["auto_flag"] == "up").sum()
    down = (df["auto_flag"] == "down").sum()
    print(f"  -> data/raw/news_auto.parquet  ({len(df)} players: "
          f"{up} up, {down} down)")
    print("  now run: .venv/bin/python run_pipeline.py --stage spreadsheet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
