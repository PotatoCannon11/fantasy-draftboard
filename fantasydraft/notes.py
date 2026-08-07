"""The Notes column, shared by every surface.

This lives on its own because `notes` and `news_flag` are NOT stored in
draft_board.parquet - they are derived at render time from three separate
sources, two of which you edit during draft week. Any surface that skips this
step shows an empty Notes column and silently drops the whole news pass.

Kept deliberately cheap and side-effect free so a dashboard can call it on a
keypress to pick up a fresh `config/news.yaml` mid-draft.
"""
from __future__ import annotations

import pandas as pd

from common import load_news, norm_name
from news_auto import load_auto_news

# Injury designations severe enough to auto-flag a note, and their colour.
INJ_DOWN = {"IR", "PUP", "NFI", "SUSP", "OUT", "DOUBTFUL", "DNR"}
INJ_WATCH = {"QUESTIONABLE"}


def apply_news(out: pd.DataFrame) -> pd.DataFrame:
    """Populate `notes` / `news_flag` from three channels, in priority order:
      1. auto injury designation already on the board (IR/PUP/OUT/...),
      2. manual notes from config/news.yaml (your curated draft-day intel),
      3. automated buzz + reports (Sleeper trending + ESPN), cached by
         news_auto - fills the cells you have not hand-written.
    Curated channels (1-2) always win; the automated channel only fills gaps so
    it never buries a note you wrote. The resulting flag drives the row colour.
    """
    news = load_news()
    auto = load_auto_news()
    notes, flags = [], []
    for _, row in out.iterrows():
        inj = str(row.get("injury_status") or "").strip().upper()
        key = norm_name(row.get("player_name"))
        manual = news.get(key, {})
        auto_n = auto.get(key, {})
        note_bits = []
        flag = manual.get("flag")
        if inj:
            note_bits.append(inj)
            if flag is None:
                flag = "out" if inj in INJ_DOWN else (
                    "watch" if inj in INJ_WATCH else None)
        if manual.get("note"):
            note_bits.append(manual["note"])
        # Automated channel: only add if this player has no curated note yet.
        if not note_bits and auto_n.get("note"):
            note_bits.append(auto_n["note"])
            if flag is None:
                flag = auto_n.get("flag")
        notes.append(" - ".join(note_bits))
        flags.append(flag or "")
    out["notes"] = notes
    out["news_flag"] = flags
    return out
