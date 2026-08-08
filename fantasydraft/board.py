"""Assembles the final draft board: projections + variance + VBD + tiers + ADP."""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from common import DATA_PROC, DATA_RAW, ensure_dirs, load_config, norm_pos, norm_team
from idmap import IdResolver, resolve_frame
from tiering import (
    add_tiers, add_vbd, blend_market, finalize_ranks, resolve_replacement,
)
from variance import add_variance


def load_adp(resolver: IdResolver) -> pd.DataFrame:
    path = DATA_RAW / "adp.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["player_uid"])
    adp = pd.read_parquet(path).rename(columns={"name": "player_name"})
    adp["position"] = adp["position"].map(norm_pos)
    adp["team"] = adp["team"].map(norm_team)
    adp = resolve_frame(adp, resolver)

    wide = None
    for teams, grp in adp.groupby("teams"):
        g = (grp.sort_values("adp")
                .drop_duplicates("player_uid")
                .set_index("player_uid")[["adp", "stdev", "high", "low", "bye"]])
        g.columns = [f"adp_{teams}", f"adp_sd_{teams}", f"adp_hi_{teams}",
                     f"adp_lo_{teams}", "bye_adp"]
        wide = g if wide is None else wide.join(g, how="outer", rsuffix="_dup")
    if wide is None:
        return pd.DataFrame(columns=["player_uid"])
    wide = wide.loc[:, ~wide.columns.str.endswith("_dup")]
    return wide.reset_index()


# Roster-status designations worth auto-flagging for a DRAFT, most severe
# first. QUESTIONABLE is deliberately excluded: it is a weekly in-season game
# status, and in the offseason it is stale noise (60+ players carry it in July).
# A genuinely worrying QUESTIONABLE player can still be noted by hand in
# config/news.yaml.
_INJ_SEVERITY = ["IR", "PUP", "NFI", "SUSP", "OUT", "DOUBTFUL", "DNR"]
_INJ_ALIASES = {"O": "OUT", "D": "DOUBTFUL", "ACTIVE": "", "NA": "",
                "NONE": "", "QUESTIONABLE": "", "Q": ""}


def load_injuries(resolver: IdResolver) -> pd.DataFrame:
    """Consolidate the injury designations already pulled from ESPN and Sleeper
    into one column per player. ESPN uses ACTIVE/QUESTIONABLE/OUT; Sleeper adds
    PUP/IR/DNR. We keep the most severe of the two."""
    frames = []
    for name, id_col in (("proj_espn", "espn_id"), ("proj_sleeper", "sleeper_id")):
        path = DATA_RAW / f"{name}.parquet"
        if not path.exists() or "injury_status" not in pd.read_parquet(path).columns:
            continue
        raw = pd.read_parquet(path)
        kw = {"espn_col": id_col} if name == "proj_espn" else {"sleeper_col": id_col}
        d = resolve_frame(raw[[id_col, "player_name", "position", "team",
                               "injury_status"]], resolver, **kw)
        d["injury_status"] = (d["injury_status"].astype(str).str.upper().str.strip()
                              .map(lambda s: _INJ_ALIASES.get(s, s)))
        frames.append(d[["player_uid", "injury_status"]])
    if not frames:
        return pd.DataFrame(columns=["player_uid", "injury_status"])
    allf = pd.concat(frames, ignore_index=True)
    rank = {s: i for i, s in enumerate(_INJ_SEVERITY)}
    allf["sev"] = allf["injury_status"].map(lambda s: rank.get(s, 99))
    best = (allf[allf["injury_status"].isin(_INJ_SEVERITY)]
            .sort_values("sev").drop_duplicates("player_uid"))
    return best[["player_uid", "injury_status"]]


def build(cfg: dict) -> pd.DataFrame:
    ensure_dirs()
    proj_path = DATA_PROC / "projections.parquet"
    if not proj_path.exists():
        raise SystemExit("projections.parquet missing - run project.py first")
    df = pd.read_parquet(proj_path)

    print("=== Layers 4-5: variance, VBD, tiering ===")
    resolver = IdResolver()
    adp_raw = pd.read_parquet(DATA_RAW / "adp.parquet") if (
        DATA_RAW / "adp.parquet").exists() else None
    if adp_raw is not None:
        adp_raw = adp_raw.copy()
        adp_raw["position"] = adp_raw["position"].map(norm_pos)

    df = add_variance(df, cfg)
    df = add_vbd(df, cfg, adp_raw)

    adp = load_adp(resolver)
    if not adp.empty:
        df = df.merge(adp, on="player_uid", how="left")
        matched = df[f"adp_{cfg['output']['adp_teams']}"].notna().sum()
        print(f"  ADP matched       {matched:>5} players")

    # Shrink toward the market, then rebuild ranks and cut tiers from the
    # blended score - ordering, tiers and VONA all have to share one currency.
    w = float(cfg["vbd"].get("market_blend", 0.0) or 0.0)
    if w > 0:
        df = blend_market(df, cfg)
        df = finalize_ranks(df, cfg)
        moved = (df["vbd_score"] - df[f"vbd_model_{cfg['league']['primary_team_count']}"]).abs()
        print(f"  market blend      w={w:.2f}, mean shift "
              f"{moved.mean():.1f} VBD pts")
    else:
        df = blend_market(df, cfg)
    df = add_tiers(df, cfg)

    inj = load_injuries(resolver)
    if not inj.empty:
        df = df.merge(inj, on="player_uid", how="left")
        flagged = df["injury_status"].fillna("").ne("").sum()
        print(f"  injury flags      {flagged:>5} players")

    n_teams = cfg["output"]["adp_teams"]
    adp_col = f"adp_{n_teams}"
    if adp_col in df.columns:
        # My rank expressed on the same scale as ADP so the delta is in picks.
        df["my_pick"] = df["vbd_score"].rank(ascending=False, method="min")
        df["value_vs_adp"] = df[adp_col] - df["my_pick"]
        df["value_rounds"] = df["value_vs_adp"] / n_teams
    df["bye_week"] = df.get("bye_week", pd.Series(np.nan, index=df.index))
    if "bye_adp" in df.columns:
        df["bye_week"] = df["bye_week"].fillna(df["bye_adp"])

    df = df.sort_values("vbd_score", ascending=False).reset_index(drop=True)
    df.to_parquet(DATA_PROC / "draft_board.parquet", index=False)

    for n in cfg["league"]["team_counts"]:
        ranks, method = resolve_replacement(cfg, n, adp_raw)
        pretty = ", ".join(f"{p}{int(round(r))}" for p, r in ranks.items())
        print(f"  replacement level ({n:>2}-team, {method}): {pretty}")
    print(f"  tiers: " + ", ".join(
        f"{p}={int(g['tier'].max())}" for p, g in df.groupby('position')))
    print(f"  -> data/processed/draft_board.parquet ({len(df)} rows)")
    return df


def main(argv=None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Layers 4-5 - build the draft board")
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args(argv)
    df = build(cfg)
    if args.show:
        cols = ["player_name", "position", "team", "tier", "sub_tier_label",
                "final_projection", "vbd_score",
                f"adp_{cfg['output']['adp_teams']}", "value_rounds"]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].head(args.show).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
