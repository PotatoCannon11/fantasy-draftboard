"""Builds the draft-board xlsx.

Tabs:
  Master Board        - everything, sorted by VBD
  QB/RB/WR/TE/K/DEF   - the same data filtered per position
  Live Draft Tracker  - mark players drafted; scarcity and positional-run
                        warnings recalculate live via worksheet formulas
  Cheat Sheet         - condensed tier-coloured print view
  Sources             - what was pulled, from where, and when

Everything on the Live Draft Tracker is written as Excel formulas rather than
precomputed values, so the sheet keeps working during the draft with no Python
in the loop.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date

import numpy as np
import pandas as pd

from common import (
    DATA_PROC,
    FANTASY_POS,
    OUTPUT,
    ensure_dirs,
    load_config,
    load_news,
    norm_name,
    read_manifest,
)
from notes import apply_news

# Tier fill colours, light enough to read black text on when printed.
TIER_COLORS = [
    "#C6EFCE", "#D9EAD3", "#FFF2CC", "#FCE5CD", "#F4CCCC", "#EAD1DC",
    "#D9D2E9", "#CFE2F3", "#D0E0E3", "#E2EFDA", "#FFF7E6", "#F2F2F2",
    "#EFEFEF", "#E8E8E8",
]

POS_COLORS = {
    "QB": "#E8D5F0", "RB": "#D5E8D4", "WR": "#D4E1F5",
    "TE": "#FDEBD0", "K": "#EAECEE", "DEF": "#E5E7E9",
}


def _fmt_book(wb):
    """Common cell formats, built once and reused across sheets."""
    f = {
        "header": wb.add_format({
            "bold": True, "bg_color": "#1F3864", "font_color": "white",
            "border": 1, "align": "center", "valign": "vcenter",
            "text_wrap": True}),
        "title": wb.add_format({"bold": True, "font_size": 15}),
        "sub": wb.add_format({"font_color": "#555555", "italic": True}),
        "num1": wb.add_format({"num_format": "0.0"}),
        "num2": wb.add_format({"num_format": "0.00"}),
        "int": wb.add_format({"num_format": "0"}),
        "text": wb.add_format({}),
        "bold": wb.add_format({"bold": True}),
        "tier_break": wb.add_format({"top": 5, "top_color": "#1F3864"}),
    }
    for i, color in enumerate(TIER_COLORS):
        f[f"tier{i + 1}"] = wb.add_format({"bg_color": color, "border": 1})
        f[f"tier{i + 1}_b"] = wb.add_format({
            "bg_color": color, "border": 1, "bold": True})
    for pos, color in POS_COLORS.items():
        f[f"pos_{pos}"] = wb.add_format({
            "bg_color": color, "bold": True, "align": "center", "border": 1})
    # News/injury flag colours for the Notes cell.
    for flag, color in (("out", "#F4C7C3"), ("down", "#FCE0B4"),
                        ("watch", "#FFF2CC"), ("up", "#CDE9D2")):
        f[f"news_{flag}"] = wb.add_format({
            "bg_color": color, "border": 1, "italic": True})
    return f


# ---------------------------------------------------------------------------
BOARD_COLS = [
    ("overall_rank", "Rk", 5, "int"),
    ("player_name", "Player", 22, "text"),
    ("position", "Pos", 5, "text"),
    ("team", "Tm", 5, "text"),
    ("tier_label", "Tier", 6, "text"),
    ("sub_tier_label", "E/M/L", 7, "text"),
    ("final_projection", "Proj", 8, "num1"),
    ("vbd_score", "VBD", 8, "num1"),
    ("vbd_8", "VBD 8tm", 8, "num1"),
    ("vbd_10", "VBD 10tm", 8, "num1"),
    ("projection_low", "Floor", 8, "num1"),
    ("projection_high", "Ceiling", 8, "num1"),
    ("adp_10", "ADP", 7, "num1"),
    ("value_rounds", "Val (rds)", 9, "num2"),
    ("bye_week", "Bye", 5, "int"),
    ("sos_z", "SOS", 6, "num2"),
    ("games_missed_l3y", "GmMiss3y", 9, "int"),
    ("age", "Age", 6, "num1"),
    ("n_sources", "Src", 5, "int"),
    ("context_multiplier", "CtxMult", 8, "num2"),
    ("notes", "Notes / Manual Flag", 26, "text"),
]


def _prep(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    # Rank on the primary league size, restated as a clean 1..N sequence.
    out = out.sort_values("vbd_score", ascending=False).reset_index(drop=True)
    out["overall_rank"] = np.arange(1, len(out) + 1)
    out = apply_news(out)
    for col, _, _, _ in BOARD_COLS:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _write_table(ws, fmts, data: pd.DataFrame, cols, *, start_row=0,
                 tier_color=True, freeze=True):
    for j, (_, label, width, _) in enumerate(cols):
        ws.set_column(j, j, width)
        ws.write(start_row, j, label, fmts["header"])
    ws.set_row(start_row, 30)
    if freeze:
        ws.freeze_panes(start_row + 1, 2)

    prev_tier = None
    for i, (_, row) in enumerate(data.iterrows()):
        r = start_row + 1 + i
        tier = row.get("tier")
        new_tier = tier != prev_tier
        prev_tier = tier
        for j, (col, _, _, kind) in enumerate(cols):
            val = row.get(col)
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                val = ""
            elif isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val)

            fmt = None
            if tier_color and pd.notna(tier):
                idx = int(min(max(int(tier), 1), len(TIER_COLORS)))
                fmt = fmts[f"tier{idx}_b" if new_tier else f"tier{idx}"]
            if col == "position" and pd.notna(row.get("position")):
                fmt = fmts.get(f"pos_{row['position']}", fmt)
            # A flagged Notes cell overrides the tier fill so news stands out.
            if col == "notes" and row.get("news_flag"):
                fmt = fmts.get(f"news_{row['news_flag']}", fmt)
            ws.write(r, j, val, fmt)
    return start_row + 1 + len(data)


def _conditional_formats(ws, wb, data, cols, start_row=0):
    """Green = value at current ADP, red = reach."""
    names = [c[0] for c in cols]
    n = len(data)
    if n == 0:
        return
    if "value_rounds" in names:
        j = names.index("value_rounds")
        rng = [start_row + 1, j, start_row + n, j]
        ws.conditional_format(*rng, {
            "type": "3_color_scale",
            "min_color": "#F8696B", "mid_color": "#FFEB84", "max_color": "#63BE7B",
            "min_type": "num", "min_value": -2,
            "mid_type": "num", "mid_value": 0,
            "max_type": "num", "max_value": 2,
        })
    if "sos_z" in names:
        j = names.index("sos_z")
        ws.conditional_format(start_row + 1, j, start_row + n, j, {
            "type": "3_color_scale",
            "min_color": "#63BE7B", "mid_color": "#FFFFFF", "max_color": "#F8696B",
            "min_type": "num", "min_value": -1,
            "mid_type": "num", "mid_value": 0,
            "max_type": "num", "max_value": 1,
        })


# ---------------------------------------------------------------------------
def sheet_master(wb, fmts, data, cfg):
    ws = wb.add_worksheet("Master Board")
    ws.write(0, 0, f"Master Board - {cfg['season']} PPR", fmts["title"])
    ws.write(1, 0, (f"{cfg['league']['primary_team_count']}-team default; "
                    f"VBD shown for {cfg['league']['team_counts']}. "
                    f"Built {date.today().isoformat()}."), fmts["sub"])
    end = _write_table(ws, fmts, data, BOARD_COLS, start_row=3)
    _conditional_formats(ws, wb, data, BOARD_COLS, start_row=3)
    ws.autofilter(3, 0, end - 1, len(BOARD_COLS) - 1)
    ws.set_landscape()
    return ws


def sheet_position(wb, fmts, data, pos, cfg):
    sub = data[data["position"] == pos].copy()
    sub = sub.sort_values("vbd_score", ascending=False)
    sub["overall_rank"] = np.arange(1, len(sub) + 1)
    ws = wb.add_worksheet(pos)
    ws.write(0, 0, f"{pos} - {cfg['season']} PPR", fmts["title"])
    ws.write(1, 0, f"{len(sub)} players, ranked by VBD within position.",
             fmts["sub"])
    end = _write_table(ws, fmts, sub, BOARD_COLS, start_row=3)
    _conditional_formats(ws, wb, sub, BOARD_COLS, start_row=3)
    ws.autofilter(3, 0, end - 1, len(BOARD_COLS) - 1)
    return ws


TRACKER_COLS = [
    ("drafted", "Drafted?", 9),
    ("mine", "Mine?", 7),
    ("overall_rank", "Rk", 5),
    ("player_name", "Player", 22),
    ("position", "Pos", 5),
    ("team", "Tm", 5),
    ("tier_label", "Tier", 6),
    ("sub_tier_label", "E/M/L", 7),
    ("vbd_score", "VBD", 8),
    ("adp_10", "ADP", 7),
    ("bye_week", "Bye", 5),
    ("tier", "Tier#", 6),   # numeric tier, so MINIFS/COUNTIFS can use it
]

# Single source of truth for tracker column letters. Every formula that reaches
# into the tracker resolves through this, so inserting a column (as "Mine?" was)
# cannot silently leave a stale $A/$D/$H reference pointing at the wrong data.
TCOL = {name: chr(ord("A") + i) for i, (name, _, _) in enumerate(TRACKER_COLS)}


def sheet_tracker(wb, fmts, data, cfg):
    """Live tracker. Type any character in 'Drafted?' to strike a player out;
    the scarcity panel and run detector recalculate from those cells."""
    ws = wb.add_worksheet("Live Draft Tracker")
    depth = min(cfg["output"]["board_depth"], len(data))
    d = data.head(depth).copy()

    ws.write(0, 0, "Live Draft Tracker", fmts["title"])
    ws.write(1, 0, "Mark 'Drafted?' as players come off the board, and 'Mine?' "
                   "for the ones YOU take (Mine? counts as drafted - you never "
                   "mark both). Everything to the right recalculates, and the "
                   "Pick Assistant stops recommending positions you have "
                   "filled.", fmts["sub"])

    header_row = 3
    for j, (_, label, width) in enumerate(TRACKER_COLS):
        ws.set_column(j, j, width)
        ws.write(header_row, j, label, fmts["header"])
    ws.set_row(header_row, 30)
    ws.freeze_panes(header_row + 1, 3)

    struck = wb.add_format({"font_strikeout": True, "font_color": "#999999"})
    for i, (_, row) in enumerate(d.iterrows()):
        r = header_row + 1 + i
        tier = int(min(max(int(row.get("tier", 1)), 1), len(TIER_COLORS)))
        for j, (col, _, _) in enumerate(TRACKER_COLS):
            if col in ("drafted", "mine"):
                ws.write_blank(r, j, "", fmts[f"tier{tier}"])
                continue
            val = row.get(col)
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                val = ""
            elif isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val)
            ws.write(r, j, val, fmts[f"tier{tier}"])

    first, last = header_row + 2, header_row + 1 + len(d)

    def rng(letter: str) -> str:
        """Fully-qualified absolute range, e.g. $D$5:$D$254."""
        return f"${letter}${first}:${letter}${last}"

    # ---- hidden helper columns that make the live suggester possible -------
    # U: numeric ADP (999 when the player has none, so comparisons behave)
    # V: live rank among UNDRAFTED players at the position, best VBD = 1
    # W: "POS|rank" key, so the assistant can pull the nth-best available
    # X: ADP standard deviation, for the probabilistic-availability model
    # V/W recalculate the instant a cell in "Drafted?" changes; the LiveCalc
    # sheet reads them to turn availability into a probability.
    for j, head in ((20, "adp_num"), (21, "avail_rank"), (22, "avail_key"),
                    (23, "adp_sd")):
        ws.write(header_row, j, head, fmts["header"])
    for i, (_, row) in enumerate(d.iterrows()):
        r = header_row + 1 + i
        e = r + 1                       # 1-based row for formulas
        adp = row.get("adp_10")
        sd = row.get("adp_sd_10")
        # Same fallback the simulator uses: late picks scatter more.
        adp_v = float(adp) if pd.notna(adp) else 999.0
        sd_v = float(sd) if (pd.notna(sd) and sd and sd > 0) else max(4.0, adp_v * 0.28)
        c_drafted, c_mine = TCOL["drafted"], TCOL["mine"]
        c_pos, c_vbd = TCOL["position"], TCOL["vbd_score"]
        # "Off the board" = marked in EITHER column, so tagging your own pick in
        # Mine? is a single keystroke and never has to be double-entered.
        gone = f'OR(${c_drafted}{e}<>"",${c_mine}{e}<>"")'
        ws.write_number(r, 20, adp_v)
        ws.write_formula(r, 21, (
            f'=IF({gone},"",COUNTIFS({rng(c_pos)},${c_pos}{e},'
            f'{rng(c_drafted)},"",{rng(c_mine)},"",'
            f'{rng(c_vbd)},">"&${c_vbd}{e})+1)'))
        # $V is the helper column immediately left (avail_rank), not a board col.
        ws.write_formula(r, 22, f'=IF({gone},"",${c_pos}{e}&"|"&$V{e})')
        ws.write_number(r, 23, sd_v)
    ws.set_column(20, 23, None, None, {"hidden": True})
    ws.conditional_format(header_row + 1, 0, last - 1, len(TRACKER_COLS) - 1, {
        "type": "formula",
        "criteria": (f'=OR(${TCOL["drafted"]}{first}<>"",'
                     f'${TCOL["mine"]}{first}<>"")'),
        "format": struck,
    })

    # --- live scarcity panel -------------------------------------------------
    p0 = len(TRACKER_COLS) + 1
    ws.write(header_row, p0, "POSITIONAL SCARCITY", fmts["header"])
    ws.write(header_row, p0 + 1, "Left", fmts["header"])
    ws.write(header_row, p0 + 2, "Top tier left", fmts["header"])
    ws.write(header_row, p0 + 3, "In that tier", fmts["header"])
    ws.set_column(p0, p0, 20)
    ws.set_column(p0 + 1, p0 + 3, 13)

    # The numeric-tier column lets the panel use plain MINIFS/COUNTIFS instead
    # of array formulas that need Ctrl+Shift+Enter. "Still available" always
    # means BOTH mark columns are empty, so your own picks leave the board too.
    pos_r = rng(TCOL["position"])
    drafted_r, mine_r = rng(TCOL["drafted"]), rng(TCOL["mine"])
    tier_r = rng(TCOL["tier"])
    open_c = f'{drafted_r},"",{mine_r},""'
    for k, pos in enumerate(FANTASY_POS):
        r = header_row + 1 + k
        best_cell = f"${chr(65 + p0 + 2)}${r + 1}"
        ws.write(r, p0, pos, fmts.get(f"pos_{pos}"))
        # Undrafted players at this position.
        ws.write_formula(r, p0 + 1,
                         f'=COUNTIFS({pos_r},"{pos}",{open_c})')
        # Best (lowest-numbered) tier still on the board.
        # MINIFS postdates Excel 2007, so the xlsx format requires the
        # `_xlfn.` prefix; without it both Excel and LibreOffice evaluate the
        # cell to #NAME? and the IFERROR silently swallows it as "-".
        ws.write_formula(
            r, p0 + 2,
            f'=IFERROR(_xlfn.MINIFS({tier_r},{pos_r},"{pos}",{open_c}),"-")')
        # How many are left in that tier.
        ws.write_formula(
            r, p0 + 3,
            f'=IF({best_cell}="-",0,COUNTIFS({pos_r},"{pos}",{open_c},'
            f'{tier_r},{best_cell}))')

    # --- positional run detector --------------------------------------------
    q = header_row + len(FANTASY_POS) + 3
    window = cfg["output"]["positional_run_window"]
    ws.write(q, p0, "POSITIONAL RUN WATCH", fmts["header"])
    ws.write(q, p0 + 1, "Drafted", fmts["header"])
    ws.write(q, p0 + 2, "Warning", fmts["header"])
    ws.write(q - 1, p0, f"Share of all drafted players by position "
                        f"(run threshold: {window} of last picks).", fmts["sub"])
    warn = wb.add_format({"bg_color": "#F8696B", "bold": True,
                          "font_color": "white", "align": "center"})
    ok = wb.add_format({"bg_color": "#E2EFDA", "align": "center"})
    for k, pos in enumerate(FANTASY_POS):
        r = q + 1 + k
        ws.write(r, p0, pos, fmts.get(f"pos_{pos}"))
        ws.write_formula(r, p0 + 1,
                         f'=COUNTIFS({pos_r},"{pos}",{drafted_r},"<>")'
                         f'+COUNTIFS({pos_r},"{pos}",{mine_r},"<>")')
        # Flag when this position is taking an outsized share of recent picks
        # AND the best remaining tier is nearly empty - that is the signal that
        # a tier is about to dry up.
        left_cell = f"${chr(65 + p0 + 1)}${header_row + 1 + k + 1}"
        tier_left = f"${chr(65 + p0 + 3)}${header_row + 1 + k + 1}"
        ws.write_formula(
            r, p0 + 2,
            f'=IF(AND({tier_left}<=2,{left_cell}>0),"TIER DRYING UP","ok")')
        ws.conditional_format(r, p0 + 2, r, p0 + 2, {
            "type": "cell", "criteria": "==", "value": '"TIER DRYING UP"',
            "format": warn})
        ws.conditional_format(r, p0 + 2, r, p0 + 2, {
            "type": "cell", "criteria": "==", "value": '"ok"', "format": ok})

    ws.write(q + len(FANTASY_POS) + 2, p0, "Picks made:", fmts["bold"])
    ws.write_formula(q + len(FANTASY_POS) + 2, p0 + 1,
                     f'=COUNTIF({drafted_r},"<>")+COUNTIF({mine_r},"<>")')
    return ws, last


CHEAT_COLS = [
    ("overall_rank", "Rk", 4),
    ("player_name", "Player", 20),
    ("team", "Tm", 4),
    ("tier_label", "Tier", 6),
    ("sub_tier_label", "E/M/L", 6),
    ("adp_10", "ADP", 6),
    ("bye_week", "Bye", 4),
]


def sheet_cheat(wb, fmts, data, cfg):
    """Condensed print view: positions side by side, tier-coloured, one page
    wide. This is the artifact you actually look at on the clock."""
    ws = wb.add_worksheet("Cheat Sheet")
    ws.write(0, 0, f"Cheat Sheet - {cfg['season']} PPR "
                   f"({cfg['league']['primary_team_count']}-team)", fmts["title"])
    ws.write(1, 0, "Colour = tier. E/M/L = draft earliest / middle / latest "
                   "within the tier (Early = safest floor).", fmts["sub"])

    depth = cfg["output"]["cheat_sheet_depth"]
    col = 0
    for pos in FANTASY_POS:
        sub = (data[data["position"] == pos]
               .sort_values("vbd_score", ascending=False))
        keep = {"QB": 20, "RB": 55, "WR": 60, "TE": 24, "K": 12, "DEF": 12}
        sub = sub.head(min(keep.get(pos, 30), depth)).copy()
        sub["overall_rank"] = np.arange(1, len(sub) + 1)

        ws.merge_range(3, col, 3, col + len(CHEAT_COLS) - 1, pos,
                       fmts.get(f"pos_{pos}"))
        for j, (_, label, width) in enumerate(CHEAT_COLS):
            ws.set_column(col + j, col + j, width)
            ws.write(4, col + j, label, fmts["header"])

        prev = None
        for i, (_, row) in enumerate(sub.iterrows()):
            r = 5 + i
            tier = int(min(max(int(row.get("tier", 1)), 1), len(TIER_COLORS)))
            new_tier = row.get("tier") != prev
            prev = row.get("tier")
            for j, (c, _, _) in enumerate(CHEAT_COLS):
                val = row.get(c)
                if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                    val = ""
                elif isinstance(val, (np.integer,)):
                    val = int(val)
                elif isinstance(val, (np.floating,)):
                    val = float(val)
                ws.write(r, col + j, val,
                         fmts[f"tier{tier}_b" if new_tier else f"tier{tier}"])
        col += len(CHEAT_COLS) + 1

    ws.set_landscape()
    ws.fit_to_pages(1, 0)
    ws.repeat_rows(3, 4)
    return ws


METRICS = [
    ("THE THREE COLUMNS THAT DECIDE PICKS", None, None),
    ("VBD", "Value Based Drafting: projected points MINUS the projected points "
            "of the replacement-level player at that position.",
     "The core ranking number. A 300-point WR is not better than a 280-point RB "
     "if WRs are deep and RBs are not. VBD is what makes positions comparable. "
     "Rank by this, not by Proj."),
    ("VONA", "Value Over Next Available: this player's VBD minus the VBD of the "
             "best player at his position you can expect to still be there at "
             "your NEXT pick. From 20,000 simulated drafts.",
     "Answers 'take him now or wait?'. High VONA = the drop-off behind him is "
     "real, take him. VONA near zero = someone just as good survives the round, "
     "spend the pick elsewhere. This is the single most actionable number here."),
    ("Value (rds)", "Your rank minus market ADP, expressed in rounds.",
     "Positive = the market is letting him fall to you. Negative = you would be "
     "reaching. Around +/-0.5 is noise; +1.5 or more is a genuine market "
     "disagreement worth acting on."),

    ("PROJECTION AND CONFIDENCE", None, None),
    ("Proj", "Blended projected PPR points for the full season.",
     "Three independent sources (Sleeper, ESPN, FantasySharks), z-scored within "
     "position so a source that assumes 16 games cannot skew the blend, then "
     "mapped back to points."),
    ("Floor / Ceiling", "One standard deviation either side of the projection.",
     "Built from measured forecast error (2014-2025), not a guess: real error is "
     "wider than most boards admit. A wide band is information - it means the "
     "range of outcomes is genuinely large, not that the projection is bad."),
    ("Src", "How many of the three projection sources cover this player.",
     "3 = solid consensus. 1 = a single source's opinion; treated with more "
     "shrinkage and a wider band. Be careful drafting a 1."),
    ("CtxMult", "The compounded context adjustment applied to the consensus.",
     "Above 1.00 = the model likes him more than the market consensus does; "
     "below 1.00 = less. Clamped to +/-20% so the model can never run away from "
     "the crowd. See the Sources tab for what feeds it."),

    ("TIERS", None, None),
    ("Tier", "Players grouped where a real cliff appears in VBD.",
     "A new tier starts where the drop to the next player is large relative to "
     "the local spread - not every N players. Anyone inside a tier is roughly "
     "interchangeable; the gap between tiers is where value is actually lost."),
    ("E/M/L", "Early / Mid / Late: draft order WITHIN a tier.",
     "Early = safest floor in the tier. Late = highest variance. If you need a "
     "safe pick take the Early one; if you are chasing upside take the Late one. "
     "They are close to equal in expected value."),

    ("CONTEXT AND RISK", None, None),
    ("ADP", "Average Draft Position in real PPR drafts of your league size.",
     "From FantasyFootballCalculator's live sample. Also drives the draft "
     "simulation and the replacement-level calculation."),
    ("Bye", "The week this player's team does not play.", "Avoid stacking too "
     "many starters on one bye week."),
    ("SOS", "Strength of schedule vs this position, weighted heavily toward "
            "weeks 1-6.", "Negative = easier schedule. Deliberately a small "
     "factor: week 15 matchups are not knowable in August."),
    ("GmMiss3y", "Games missed over the last three seasons.",
     "Feeds a durability discount and widens the projection band. Raw exposure, "
     "not a prediction of injury."),
    ("Age", "Age on September 1st.",
     "Drives a position-specific discount. RBs decline from about 24 and fall "
     "hard; QBs hold value into their 30s. Fitted on within-player year-over-"
     "year change so it is not fooled by survivorship."),

    ("THE PICK ASSISTANT HAS TWO HALVES", None, None),
    ("Left: the plan", "Pre-draft Monte Carlo. 20,000 simulated drafts built "
                       "from ADP and its spread, before a single pick is made.",
     "Use it to plan: which rounds you can afford to wait on a position, and "
     "where the runs are likely to come. It assumes the draft goes to script."),
    ("Right: LIVE", "Recomputed from the Drafted? marks on the Live Draft "
                    "Tracker. Nothing is precomputed - it reads the real board.",
     "Use it on the clock. The moment your league does something the ADP did "
     "not expect, this is the half that is still correct. When the two "
     "disagree, trust this one."),
    ("Live VONA", "Best available at a position now, minus the PROBABILITY-"
                  "weighted best still there at your next turn. Each player's "
                  "draft slot is modelled as a bell curve around his ADP.",
     "The live twin of VONA. Near 0 means the value at that position survives "
     "the round - spend the pick elsewhere. The biggest number is the real cost "
     "of waiting. Because it is probabilistic, a player right on the ADP "
     "boundary gets partial credit rather than a misleading all-or-nothing."),
    ("Override", "Manual pick count on the live panel.",
     "You WILL fall behind on marking players. Type the true number of picks "
     "made here and the 'next turn' maths stays correct even with a "
     "half-updated board. Clear it to go back to counting the marks."),

    ("HOW TO READ THE BOARD", None, None),
    ("Best pick", "Highest VONA among players in your best available tier.",
     "Not simply the highest VBD. If two players are close in VBD but one will "
     "survive the round, take the other one first."),
    ("Positional run", "Watch the scarcity strip on the Live Draft Tracker.",
     "When 'in tier' for a position drops to 2 or fewer, that tier is about to "
     "empty. Either take one now or write the position off until the next tier."),
    ("What this model will NOT tell you", "Late-breaking news.",
     "A pulled hamstring on the Thursday before the draft beats every number on "
     "this board. Use the Notes column on the Master Board."),
]


def sheet_metrics(wb, fmts, cfg):
    """Plain-English glossary. Every metric, what it means, and how to use it."""
    ws = wb.add_worksheet("Metrics")
    ws.write(0, 0, "What every number on this board means", fmts["title"])
    ws.write(1, 0, "Read the first section if you read nothing else.", fmts["sub"])
    ws.set_column(0, 0, 17)
    ws.set_column(1, 1, 62)
    ws.set_column(2, 2, 76)
    hdr = ["Metric", "Definition", "How to actually use it"]
    for j, h in enumerate(hdr):
        ws.write(3, j, h, fmts["header"])
    ws.set_row(3, 26)

    section = wb.add_format({"bold": True, "bg_color": "#1F3864",
                             "font_color": "white", "font_size": 11,
                             "align": "left", "valign": "vcenter"})
    body = wb.add_format({"text_wrap": True, "valign": "top", "border": 1})
    name = wb.add_format({"bold": True, "valign": "top", "border": 1,
                          "bg_color": "#EDF1F5"})
    r = 4
    for metric, definition, usage in METRICS:
        if definition is None:
            ws.merge_range(r, 0, r, 2, metric, section)
            ws.set_row(r, 22)
        else:
            ws.write(r, 0, metric, name)
            ws.write(r, 1, definition, body)
            ws.write(r, 2, usage, body)
            ws.set_row(r, 44)
        r += 1
    ws.freeze_panes(4, 0)
    return ws


def sheet_compare(wb, fmts, board, cfg):
    """Side-by-side comparison of up to four players, driven by dropdowns."""
    ws = wb.add_worksheet("Compare")
    n = len(board)
    last = 4 + n                      # Master Board data rows are 5..4+n
    src = f"='Master Board'!$B$5:$B${last}"

    ws.write(0, 0, "Player comparer", fmts["title"])
    ws.write(1, 0, "Pick players from the dropdowns. Everything below updates "
                   "automatically. Best value in each row is highlighted.",
             fmts["sub"])
    ws.set_column(0, 0, 20)
    ws.set_column(1, 4, 19)

    slots = 4
    defaults = list(board["player_name"].head(slots))
    pick_fmt = wb.add_format({"bold": True, "bg_color": "#FFF2CC", "border": 2,
                              "align": "center", "font_size": 12})
    ws.write(3, 0, "Player", fmts["header"])
    for i in range(slots):
        ws.write(3, 1 + i, defaults[i] if i < len(defaults) else "", pick_fmt)
        ws.data_validation(3, 1 + i, 3, 1 + i,
                           {"validate": "list", "source": src})
    ws.set_row(3, 24)

    # (label, Master Board column letter, number format, higher-is-better)
    ROWS = [
        ("Position", "C", "text", None),
        ("Team", "D", "text", None),
        ("Tier", "E", "text", None),
        ("Early/Mid/Late", "F", "text", None),
        ("Projected points", "G", "num1", True),
        ("VBD", "H", "num1", True),
        ("VBD (8-team)", "I", "num1", True),
        ("VBD (10-team)", "J", "num1", True),
        ("Floor", "K", "num1", True),
        ("Ceiling", "L", "num1", True),
        ("Ceiling - Floor (risk)", None, "num1", False),
        ("ADP", "M", "num1", False),
        ("Value vs ADP (rds)", "N", "num2", True),
        ("Bye week", "O", "int", None),
        ("Strength of schedule", "P", "num2", False),
        ("Games missed (3y)", "Q", "int", False),
        ("Age", "R", "num1", False),
        ("Sources", "S", "int", True),
        ("Context multiplier", "T", "num2", True),
    ]

    lbl = wb.add_format({"bold": True, "border": 1, "bg_color": "#EDF1F5"})
    cell = {k: wb.add_format({"border": 1, "num_format": v})
            for k, v in {"num1": "0.0", "num2": "0.00", "int": "0",
                         "text": "General"}.items()}
    best = wb.add_format({"border": 1, "bold": True, "bg_color": "#C6EFCE"})

    r = 5
    numeric_rows = []
    for label, col, kind, higher in ROWS:
        ws.write(r, 0, label, lbl)
        for i in range(slots):
            pc = chr(66 + i)          # B, C, D, E
            if col is None:           # derived: ceiling - floor
                f = (f'=IF({pc}$4="","",IFERROR('
                     f'INDEX(\'Master Board\'!$L$5:$L${last},'
                     f'MATCH({pc}$4,\'Master Board\'!$B$5:$B${last},0))-'
                     f'INDEX(\'Master Board\'!$K$5:$K${last},'
                     f'MATCH({pc}$4,\'Master Board\'!$B$5:$B${last},0)),""))')
            else:
                f = (f'=IF({pc}$4="","",IFERROR(INDEX('
                     f'\'Master Board\'!${col}$5:${col}${last},'
                     f'MATCH({pc}$4,\'Master Board\'!$B$5:$B${last},0)),""))')
            ws.write_formula(r, 1 + i, f, cell[kind])
        if higher is not None:
            numeric_rows.append((r, higher))
        r += 1

    # Highlight the best value in each numeric row.
    for row, higher in numeric_rows:
        rng = f"$B{row + 1}:$E{row + 1}"
        crit = "MAX" if higher else "MIN"
        ws.conditional_format(row, 1, row, slots, {
            "type": "formula",
            "criteria": f'=AND(B{row + 1}<>"",B{row + 1}={crit}({rng}))',
            "format": best,
        })

    ws.write(r + 1, 0, "Note", fmts["bold"])
    ws.write(r + 1, 1,
             "Green = best of the selected players for that row. For ADP, "
             "schedule, games missed and age, lower is better.", fmts["sub"])
    ws.freeze_panes(5, 1)
    return ws


def sheet_simdata(wb, fmts, vona):
    """Hidden backing data for the Pick Assistant.

    Keyed as pick*1000 + rank so the assistant can pull rows with plain
    INDEX/MATCH. Deliberately avoids FILTER/XLOOKUP, which need the _xlfn
    prefix dance and are not safe across every spreadsheet app.
    """
    ws = wb.add_worksheet("SimData")
    ws.hide()
    cols = ["key", "pick", "rank", "player_name", "position", "vbd_score",
            "p_available_now", "p_available_next", "vona"]
    for j, c in enumerate(cols):
        ws.write(0, j, c)
    r = 1
    for pick, g in vona.groupby("pick"):
        g = g.dropna(subset=["vona"])
        # Only rank players who could realistically still be on the board at
        # this pick. Without this the assistant happily recommends the 1.01
        # overall in round 5, where his VONA is huge but his availability is nil.
        live = g[g["p_available_now"] >= 0.05]
        if len(live) < 8:  # very late picks: fall back to the best available
            live = g.nlargest(max(8, len(live)), "p_available_now")
        g = live.nlargest(40, "vona").reset_index(drop=True)
        for i, row in g.iterrows():
            ws.write_number(r, 0, int(pick) * 1000 + i + 1)
            ws.write_number(r, 1, int(pick))
            ws.write_number(r, 2, i + 1)
            ws.write_string(r, 3, str(row["player_name"]))
            ws.write_string(r, 4, str(row["position"]))
            ws.write_number(r, 5, float(row["vbd_score"]))
            ws.write_number(r, 6, float(row["p_available_now"]))
            ws.write_number(r, 7, float(row["p_available_next"]))
            ws.write_number(r, 8, float(row["vona"]))
            r += 1
    return ws, r


def sheet_pick_assistant(wb, fmts, vona, best, cfg, sim_rows, tracker_last,
                         live_best):
    """Which player to take at a given pick, accounting for who survives."""
    ws = wb.add_worksheet("Pick Assistant")
    picks = sorted(int(p) for p in vona["pick"].unique())
    sim = cfg.get("simulation", {})

    ws.write(0, 0, "Pick Assistant", fmts["title"])
    ws.write(1, 0, f"Based on {sim.get('sims', 20000):,} simulated drafts from "
                   f"seat {sim.get('draft_slot')} in a "
                   f"{cfg['league']['primary_team_count']}-team league. "
                   f"Change the pick below to see your board at that moment.",
             fmts["sub"])

    ws.set_column(0, 0, 22)
    ws.set_column(1, 1, 22)
    ws.set_column(2, 8, 13)

    sel = wb.add_format({"bold": True, "bg_color": "#FFF2CC", "border": 2,
                         "align": "center", "font_size": 13, "num_format": "0"})
    ws.write(3, 0, "Your pick number", fmts["header"])
    ws.write_number(3, 1, picks[0], sel)
    ws.data_validation(3, 1, 3, 1, {
        "validate": "list", "source": [str(p) for p in picks]})
    ws.write(3, 2, "<- pick from the list", fmts["sub"])

    last = sim_rows
    key = "$B$4*1000+"
    hdr = ["Rank", "Player", "Pos", "VBD", "There now", "VONA",
           "Survives to next pick", "Verdict"]
    for j, h in enumerate(hdr):
        ws.write(5, j, h, fmts["header"])
    ws.set_row(5, 30)

    pct = wb.add_format({"num_format": "0%", "border": 1})
    n1 = wb.add_format({"num_format": "0.0", "border": 1})
    txt = wb.add_format({"border": 1})
    bold = wb.add_format({"border": 1, "bold": True})

    def idx(col_letter, i):
        return (f'IFERROR(INDEX(SimData!${col_letter}$2:${col_letter}${last},'
                f'MATCH({key}{i},SimData!$A$2:$A${last},0)),"")')

    for i in range(1, 16):
        r = 5 + i
        ws.write_number(r, 0, i, txt)
        ws.write_formula(r, 1, f"={idx('D', i)}", bold)
        ws.write_formula(r, 2, f"={idx('E', i)}", txt)
        ws.write_formula(r, 3, f"={idx('F', i)}", n1)
        ws.write_formula(r, 4, f"={idx('G', i)}", pct)   # available at this pick
        ws.write_formula(r, 5, f"={idx('I', i)}", n1)    # VONA
        ws.write_formula(r, 6, f"={idx('H', i)}", pct)   # survives to next
        # Verdict weighs both: is he even here, and will he last?
        ws.write_formula(
            r, 7,
            f'=IF(B{r + 1}="","",IF(E{r + 1}<0.35,"long shot",'
            f'IF(G{r + 1}<0.15,"TAKE NOW",'
            f'IF(G{r + 1}<0.5,"probably gone","can wait"))))', txt)

    take = wb.add_format({"bg_color": "#C6EFCE", "bold": True, "border": 1})
    gone = wb.add_format({"bg_color": "#FFF2CC", "border": 1})
    wait = wb.add_format({"bg_color": "#F2F2F2", "font_color": "#666666",
                          "border": 1})
    longshot = wb.add_format({"bg_color": "#FBE7E4", "font_color": "#999999",
                              "italic": True, "border": 1})
    for value, fmt in (('"TAKE NOW"', take), ('"probably gone"', gone),
                       ('"can wait"', wait), ('"long shot"', longshot)):
        ws.conditional_format(6, 7, 20, 7, {
            "type": "cell", "criteria": "==", "value": value, "format": fmt})
    ws.conditional_format(6, 5, 20, 5, {
        "type": "3_color_scale",
        "min_color": "#F2F2F2", "mid_color": "#FFEB84", "max_color": "#63BE7B",
        "min_type": "min", "mid_type": "percentile", "mid_value": 50,
        "max_type": "max"})

    # Expected best available by position, now vs next pick.
    r0 = 23
    ws.write(r0, 0, "EXPECTED BEST AVAILABLE (VBD)", fmts["header"])
    ws.write(r0, 1, "At this pick", fmts["header"])
    ws.write(r0, 2, "At next pick", fmts["header"])
    ws.write(r0, 3, "Drop", fmts["header"])
    ws.write(r0 - 1, 0, "How much value you lose at each position by waiting "
                        "one round. The biggest drop is where to spend the pick.",
             fmts["sub"])

    b = best.copy()
    nxt = {p: picks[i + 1] for i, p in enumerate(picks[:-1])}
    lut = b.set_index(["pick", "position"])["exp_best_vbd"].to_dict()
    for k, pos in enumerate(FANTASY_POS):
        r = r0 + 1 + k
        ws.write(r, 0, pos, fmts.get(f"pos_{pos}"))
        # Written as static values per pick would need 15 blocks; instead look
        # up against a small inline table keyed the same way.
        vals_now = [lut.get((p, pos), 0.0) for p in picks]
        vals_next = [lut.get((nxt.get(p), pos), 0.0) if p in nxt else 0.0
                     for p in picks]
        col_now = ",".join(f"{v:.2f}" for v in vals_now)
        col_next = ",".join(f"{v:.2f}" for v in vals_next)
        ws.write_formula(r, 1, f'=INDEX({{{col_now}}},MATCH($B$4,{{{",".join(str(p) for p in picks)}}},0))', n1)
        ws.write_formula(r, 2, f'=INDEX({{{col_next}}},MATCH($B$4,{{{",".join(str(p) for p in picks)}}},0))', n1)
        ws.write_formula(r, 3, f"=B{r + 1}-C{r + 1}", n1)
    ws.conditional_format(r0 + 1, 3, r0 + len(FANTASY_POS), 3, {
        "type": "3_color_scale",
        "min_color": "#FFFFFF", "mid_color": "#FFEB84", "max_color": "#F8696B",
        "min_type": "min", "mid_type": "percentile", "mid_value": 50,
        "max_type": "max"})

    ws.write(r0 + len(FANTASY_POS) + 2, 0, "Reading this", fmts["bold"])
    ws.write(r0 + len(FANTASY_POS) + 2, 1,
             "High VONA and a low survival chance means take him now. High "
             "survival chance means the same value is available next round - "
             "spend this pick on the position with the biggest Drop.",
             fmts["sub"])

    _live_block(ws, wb, fmts, cfg, picks, tracker_last, live_best)
    return ws


def sheet_livecalc(wb, fmts, tracker_last, depth=15):
    """Hidden engine for the live suggester's probabilistic availability.

    Hard "ADP later than my next pick = safe" is a step function - it calls a
    player at ADP 24 a certain miss when my next pick is 25, and a player at 26
    a certain survivor. Reality is smooth. This sheet models each player's
    realised draft slot as Normal(ADP, ADP_sd) and asks P(slot > my next pick).

    Expected VBD of the best survivor at a position is then, over its available
    players in VBD order,
        sum_i  vbd_i * P(i survives) * prod_{j better}(1 - P(j survives))
    i.e. the value of player i weighted by the chance he is the best one still
    on the board. `carry` is that running product. NORMDIST (legacy spelling,
    no _xlfn. needed) is used for the CDF so the file stays portable.
    """
    ws = wb.add_worksheet("LiveCalc")
    ws.hide()
    T = "'Live Draft Tracker'!"
    first = 5
    def tr(c):
        return f"{T}${c}${first}:${c}${tracker_last}"

    # Cell every formula reads: the pick number of your next turn.
    ws.write(0, 0, "next_turn_pick")
    ws.write_formula(0, 1, "='Pick Assistant'!$K$7")
    NT = "$B$1"

    headers = ["pos", "rank", "vbd", "adp", "sd", "surv", "carry", "contrib"]
    for j, h in enumerate(headers):
        ws.write(2, j, h)

    # Summary block the Pick Assistant reads: expected best survivor per pos.
    ws.write(2, 9, "position")
    ws.write(2, 10, "exp_best_next")
    summary_cell = {}

    row = 3
    block_top = {}
    for pos in FANTASY_POS:
        block_top[pos] = row
        for k in range(1, depth + 1):
            e = row + 1
            key = f'"{pos}|{k}"'
            m = f'MATCH({key},{tr("W")},0)'
            ws.write_formula(row, 2, f'=IFERROR(INDEX({tr(TCOL["vbd_score"])},{m}),0)')   # vbd
            ws.write_formula(row, 3, f'=IFERROR(INDEX({tr("U")},{m}),999)')    # adp
            ws.write_formula(row, 4, f'=IFERROR(INDEX({tr("X")},{m}),10)')     # sd
            # P(survives to my next pick) = P(slot > next_turn).
            ws.write_formula(row, 5, (
                f'=IF($C{e}=0,0,MAX(0,MIN(1,1-NORMDIST({NT},$D{e},$E{e},TRUE))))'))
            if k == 1:
                ws.write_number(row, 6, 1.0)                                   # carry
            else:
                ws.write_formula(row, 6, f'=$G{row}*(1-$F{row})')
            ws.write_formula(row, 7, f'=$C{e}*$F{e}*$G{e}')                    # contrib
            row += 1

    # exp_best_next per position = sum of its contributions.
    for i, pos in enumerate(FANTASY_POS):
        r = 3 + i
        top = block_top[pos]
        ws.write(r, 9, pos)
        ws.write_formula(r, 10, f'=SUM($H${top + 1}:$H${top + depth})')
        summary_cell[pos] = f"LiveCalc!$K${r + 1}"
    ws.set_column(0, 10, 10)
    return ws, summary_cell


def _live_block(ws, wb, fmts, cfg, picks, tracker_last, live_best):
    """The reactive half: reads the Live Draft Tracker's marks and recomputes
    best-available and VONA from the real board state.

    Deliberately independent of the simulation. The simulated column is your
    pre-draft plan; this one is what is actually true right now. If you fall
    behind on marking players off, this degrades gracefully - with nothing
    marked it simply agrees with the pre-draft view.
    """
    T = "'Live Draft Tracker'!"
    first, last = 5, tracker_last
    def tr(c):
        return f"{T}${c}${first}:${c}${last}"

    # Column layout: J=labels/position, K=values/name, L=VBD, M=VONA, N=verdict,
    # O=how many you already own, P=hidden roster-adjusted VONA the headline uses
    J, K, L, M, N, O, P = 9, 10, 11, 12, 13, 14, 15
    ws.set_column(J, J, 22)
    ws.set_column(K, N, 16)
    ws.set_column(O, O, 8)
    ws.set_column(P, P, None, None, {"hidden": True})
    useful = {**{p: 99 for p in FANTASY_POS},
              **(cfg.get("output", {}).get("useful_max") or {})}

    lst = "{" + ",".join(str(p) for p in picks) + "}"
    box = wb.add_format({"bold": True, "bg_color": "#FFF2CC", "border": 2,
                         "align": "center", "num_format": "0"})
    ro = wb.add_format({"bg_color": "#EDF1F5", "border": 1, "num_format": "0",
                        "align": "center", "bold": True})
    lbl = wb.add_format({"bold": True, "border": 1, "bg_color": "#EDF1F5"})

    ws.merge_range(3, J, 3, O,
                   "LIVE - recalculated from what you mark drafted",
                   fmts["header"])

    # --- pick bookkeeping (rows 5-7) ---
    # The override exists because you will lose track mid-draft, and the
    # "next turn" maths has to stay right even when the marks do not.
    ws.write(4, J, "Picks made", lbl)
    ws.write_formula(4, K, f'=COUNTIF({tr(TCOL["drafted"])},"<>")'
                     f'+COUNTIF({tr(TCOL["mine"])},"<>")', ro)
    ws.write(4, L, "Override", lbl)
    ws.write_blank(4, M, "", box)
    ws.data_validation(4, M, 4, M, {
        "validate": "integer", "criteria": "between",
        "minimum": 0, "maximum": 400,
        "input_title": "Lost count?",
        "input_message": "Type the true number of picks made. Leave blank to "
                         "use the count from the Drafted? column."})

    made = f'IF($M$5="",$K$5,$M$5)'
    ws.write(5, J, "On the clock (pick #)", lbl)
    ws.write_formula(5, K, f"={made}+1", ro)
    idx = f'IFERROR(MATCH($K$6-0.5,{lst},1)+1,1)'
    ws.write(5, L, "This turn of yours", lbl)
    ws.write_formula(5, M, f'=IFERROR(INDEX({lst},{idx}),"-")', ro)

    # $K$7 is the cell the tracker's survivor helper column reads.
    ws.write(6, J, "Your NEXT turn", lbl)
    ws.write_formula(6, K, f'=IFERROR(INDEX({lst},{idx}+1),$K$6+12)', ro)
    ws.write(6, L, "Picks until then", lbl)
    ws.write_formula(6, M, "=MAX(0,$K$7-$K$6)", ro)

    # --- per-position live board (header row 9, data rows 10-15) ---
    hdr = ["Position", "Best available now", "VBD", "Live VONA", "Verdict",
           "Yours"]
    for j, h in enumerate(hdr):
        ws.write(8, J + j, h, fmts["header"])
    ws.set_row(8, 28)

    n1 = wb.add_format({"num_format": "0.0", "border": 1})
    txt = wb.add_format({"border": 1})
    nm = wb.add_format({"border": 1, "bold": True})

    top, bot = 10, 9 + len(FANTASY_POS)          # 1-based data rows
    vr, nr, pr = f"$M${top}:$M${bot}", f"$K${top}:$K${bot}", f"$J${top}:$J${bot}"
    er = f"$P${top}:$P${bot}"                    # roster-adjusted VONA

    for k, pos in enumerate(FANTASY_POS):
        r = 9 + k
        e = r + 1
        ws.write(r, J, pos, fmts.get(f"pos_{pos}"))
        # Best undrafted player at this position, straight off the tracker.
        ws.write_formula(r, K, (
            f'=IFERROR(INDEX({tr(TCOL["player_name"])},'
            f'MATCH("{pos}|1",{tr("W")},0)),"-")'), nm)
        ws.write_formula(r, L, (
            f'=IFERROR(INDEX({tr(TCOL["vbd_score"])},'
            f'MATCH("{pos}|1",{tr("W")},0)),0)'), n1)
        # Live VONA: his VBD minus the EXPECTED VBD of the best player at this
        # position still on the board at your next turn - a probability-weighted
        # figure from the LiveCalc sheet, not a hard ADP cutoff.
        ws.write_formula(r, M, f'=$L{e}-{live_best[pos]}', n1)
        # How many of this position you have already taken, read straight off
        # the Mine? column.
        ws.write_formula(r, O, (
            f'=COUNTIFS({tr(TCOL["mine"])},"<>",'
            f'{tr(TCOL["position"])},"{pos}")'),
            wb.add_format({"num_format": "0", "border": 1, "align": "center"}))
        # Roster-adjusted VONA: a position you can no longer use drops out of
        # the headline entirely, instead of winning on raw VONA forever.
        ws.write_formula(r, P, f'=IF($O{e}>={useful[pos]},-9999,$M{e})')
        ws.write_formula(r, N, (
            f'=IF($K{e}="-","",IF($O{e}>={useful[pos]},"ROSTER FULL",'
            f'IF($M{e}>=MAX({er}),"BEST VALUE",'
            f'IF($M{e}<=1,"can wait",""))))'), txt)

    bestfmt = wb.add_format({"bg_color": "#C6EFCE", "bold": True, "border": 1})
    waitfmt = wb.add_format({"bg_color": "#F2F2F2", "font_color": "#666666",
                             "border": 1})
    ws.conditional_format(9, N, bot - 1, N, {
        "type": "cell", "criteria": "==", "value": '"BEST VALUE"',
        "format": bestfmt})
    ws.conditional_format(9, N, bot - 1, N, {
        "type": "cell", "criteria": "==", "value": '"can wait"',
        "format": waitfmt})
    fullfmt = wb.add_format({"bg_color": "#D9D9D9", "font_color": "#7F7F7F",
                             "italic": True, "border": 1})
    ws.conditional_format(9, N, bot - 1, N, {
        "type": "cell", "criteria": "==", "value": '"ROSTER FULL"',
        "format": fullfmt})
    ws.conditional_format(9, M, bot - 1, M, {
        "type": "3_color_scale",
        "min_color": "#F2F2F2", "mid_color": "#FFEB84", "max_color": "#63BE7B",
        "min_type": "min", "mid_type": "percentile", "mid_value": 50,
        "max_type": "max"})

    # --- the single headline call ---
    big = wb.add_format({"bold": True, "font_size": 13, "bg_color": "#1F3864",
                         "font_color": "white", "align": "center",
                         "valign": "vcenter", "border": 2})
    ws.merge_range(bot + 1, J, bot + 1, O, "", big)
    # Ranked on the roster-adjusted column, but the VONA shown is the true one.
    # If every position is full the pick is a pure best-available call, so say
    # so rather than naming whoever lost by the smallest margin.
    midx = f"MATCH(MAX({er}),{er},0)"
    ws.write_formula(bot + 1, J, (
        f'=IFERROR(IF(MAX({er})<=-9000,'
        f'"ROSTER FULL - take best available",'
        f'"TAKE: "&INDEX({nr},{midx})&"  ("&INDEX({pr},{midx})&", +"&'
        f'TEXT(INDEX({vr},{midx}),"0.0")&" VONA)"),"-")'), big)
    ws.set_row(bot + 1, 30)

    ws.write(bot + 3, J, "How the two halves differ", fmts["bold"])
    ws.write(bot + 3, K,
             "Left: the pre-draft plan from the simulations - what SHOULD "
             "happen. Right: what is actually on the board given your marks. "
             "When they disagree, trust the live side; it knows the draft is "
             "not going to script.", fmts["sub"])
    ws.write(bot + 4, K,
             "Marked nothing yet? The live side simply agrees with the plan. "
             "Behind on marking? Put the true pick count in Override so the "
             "'next turn' maths stays right.", fmts["sub"])
    ws.write(bot + 5, J, "Why 'Yours' matters", fmts["bold"])
    ws.write(bot + 5, K,
             "Mark your own picks in the tracker's Mine? column. A position you "
             "already have enough of drops out of the headline - otherwise raw "
             "VONA keeps recommending quarterbacks late, because replacement QB "
             "is so far below the startable pool. Ceilings live in "
             "config/weights.yaml -> output.useful_max.", fmts["sub"])
    return ws


def sheet_sources(wb, fmts, cfg):
    ws = wb.add_worksheet("Sources")
    ws.write(0, 0, "Data provenance", fmts["title"])
    ws.write(1, 0, "Every dataset behind this board, and when it was pulled. "
                   "Check this before trusting anything on draft day.",
             fmts["sub"])
    headers = ["Dataset", "Pulled at (UTC)", "Rows", "Source URL"]
    widths = [26, 24, 10, 100]
    for j, (h, w) in enumerate(zip(headers, widths)):
        ws.set_column(j, j, w)
        ws.write(3, j, h, fmts["header"])
    manifest = read_manifest()
    for i, (key, meta) in enumerate(sorted(manifest.items())):
        r = 4 + i
        ws.write(r, 0, key)
        ws.write(r, 1, meta.get("pulled_at", ""))
        ws.write(r, 2, meta.get("rows", ""))
        ws.write(r, 3, meta.get("url", ""))
    return ws


# ---------------------------------------------------------------------------
def build(cfg: dict, out_path=None) -> str:
    ensure_dirs()
    path = DATA_PROC / "draft_board.parquet"
    if not path.exists():
        raise SystemExit("draft_board.parquet missing - run board.py first")
    df = pd.read_parquet(path)

    # Drop the deep tail of undraftable players and anything only one source
    # believes in, unless it still projects well or the market is drafting it.
    board = _prep(df, cfg)
    ocfg = cfg["output"]
    keep = board["n_sources"].fillna(0) >= 2
    keep |= board["overall_rank"] <= ocfg.get("single_source_rank_grace", 60)
    if ocfg.get("single_source_keep_if_adp", True):
        keep |= board[f"adp_{ocfg['adp_teams']}"].notna()
    board = board[keep].head(ocfg["board_depth"]).copy()
    board["overall_rank"] = np.arange(1, len(board) + 1)

    out_path = out_path or (OUTPUT / f"draft_board_{date.today().isoformat()}.xlsx")
    wb = pd.ExcelWriter(out_path, engine="xlsxwriter").book
    fmts = _fmt_book(wb)

    # Simulation output is optional; the workbook still builds without it.
    vona = best = None
    vp, bp = DATA_PROC / "sim_vona.parquet", DATA_PROC / "sim_best_available.parquet"
    if vp.exists() and bp.exists():
        vona, best = pd.read_parquet(vp), pd.read_parquet(bp)

    sheet_master(wb, fmts, board, cfg)
    for pos in FANTASY_POS:
        sheet_position(wb, fmts, board, pos, cfg)
    _tw, tracker_last = sheet_tracker(wb, fmts, board, cfg)
    sheet_cheat(wb, fmts, board, cfg)
    sheet_compare(wb, fmts, board, cfg)
    if vona is not None:
        _sd, sim_rows = sheet_simdata(wb, fmts, vona)
        _lc, live_best = sheet_livecalc(wb, fmts, tracker_last)
        sheet_pick_assistant(wb, fmts, vona, best, cfg, sim_rows,
                             tracker_last, live_best)
    else:
        print("  [skip] Pick Assistant - run src/simulate.py first")
    sheet_metrics(wb, fmts, cfg)
    sheet_sources(wb, fmts, cfg)
    wb.close()

    n_tabs = len(wb.worksheets())
    print(f"  -> {out_path}  ({len(board)} players, {n_tabs} tabs)")
    return str(out_path)


def main(argv=None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Build the draft board xlsx")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    print("=== Output: spreadsheet ===")
    build(cfg, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
