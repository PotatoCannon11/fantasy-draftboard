"""Layer 2 - feature engineering.

Joins the Layer 1 raw pulls into two tables:

  data/processed/player_features.parquet  - one row per player
  data/processed/team_features.parquet    - one row per team

and (re)generates config/team_context.yaml, the hand-editable overlay of
qualitative team tags. Values there are auto-derived on first run; anything
you edit by hand is preserved on later runs.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import yaml

from common import (
    CONFIG,
    DATA_PROC,
    DATA_RAW,
    ensure_dirs,
    load_config,
    norm_pos,
    norm_team,
)
from idmap import IdResolver

SEASON_GAMES = 17


def _read(name: str) -> pd.DataFrame:
    path = DATA_RAW / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _z(s: pd.Series) -> pd.Series:
    """Z-score that degrades gracefully on constant / tiny inputs."""
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    if not sd or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return ((s - s.mean()) / sd).fillna(0.0)


def _recency_weights(seasons: list[int], half_life: float = 1.0) -> dict[int, float]:
    """Most recent season counts most; each older season is halved."""
    newest = max(seasons)
    return {s: 0.5 ** ((newest - s) / half_life) for s in seasons}


# ---------------------------------------------------------------------------
# Player bio / identity
# ---------------------------------------------------------------------------
def bio_features(resolver: IdResolver, season: int) -> pd.DataFrame:
    rosters = _read("rosters")
    players = _read("players")

    rows = []
    for r in rosters.itertuples(index=False):
        gsis = getattr(r, "gsis_id", None)
        if not gsis or str(gsis) == "nan":
            continue
        uid = resolver.canonical(gsis)
        rows.append({
            "player_uid": uid,
            "roster_team": norm_team(getattr(r, "team", None)),
            "roster_position": norm_pos(getattr(r, "position", None)),
            "birth_date": getattr(r, "birth_date", None),
            "years_exp": getattr(r, "years_exp", None),
            "rookie_year": getattr(r, "rookie_year", None),
            "entry_year": getattr(r, "entry_year", None),
            "draft_number": getattr(r, "draft_number", None),
            "roster_status": getattr(r, "status", None),
        })
    bio = pd.DataFrame(rows).drop_duplicates("player_uid")

    if not players.empty:
        p = players[["gsis_id", "birth_date", "rookie_season", "draft_round",
                     "draft_pick"]].copy()
        p["player_uid"] = p["gsis_id"].map(resolver.canonical)
        p = p.drop_duplicates("player_uid").drop(columns=["gsis_id"])
        bio = bio.merge(p, on="player_uid", how="outer", suffixes=("", "_pl"))
        bio["birth_date"] = bio["birth_date"].fillna(bio.pop("birth_date_pl"))
        bio["rookie_year"] = bio["rookie_year"].fillna(bio["rookie_season"])

    bd = pd.to_datetime(bio["birth_date"], errors="coerce", utc=True)
    season_start = pd.Timestamp(f"{season}-09-01", tz="UTC")
    bio["age"] = (season_start - bd).dt.days / 365.25

    bio["rookie_flag"] = (
        pd.to_numeric(bio["rookie_year"], errors="coerce") >= season
    ).astype(int)
    exp = pd.to_numeric(bio["years_exp"], errors="coerce")
    bio.loc[exp.eq(0) & bio["rookie_year"].isna(), "rookie_flag"] = 1
    return bio


# ---------------------------------------------------------------------------
# Depth chart
# ---------------------------------------------------------------------------
def depth_chart_features(resolver: IdResolver) -> pd.DataFrame:
    dc = _read("depth_charts")
    if dc.empty:
        return pd.DataFrame(columns=["player_uid", "depth_rank", "depth_team"])
    dc = dc[dc["pos_abb"].isin(["QB", "RB", "WR", "TE"])].copy()
    dc["player_uid"] = dc["gsis_id"].map(resolver.canonical)
    dc = dc[dc["player_uid"].notna()]
    dc["pos_rank"] = pd.to_numeric(dc["pos_rank"], errors="coerce")
    # A player can appear in several formation packages; keep the best rank.
    out = (dc.groupby("player_uid")
             .agg(depth_rank=("pos_rank", "min"),
                  depth_team=("team", "first"),
                  depth_pos=("pos_abb", "first"))
             .reset_index())
    out["depth_team"] = out["depth_team"].map(norm_team)
    return out


# ---------------------------------------------------------------------------
# Volume / opportunity from season box scores
# ---------------------------------------------------------------------------
def volume_features(resolver: IdResolver, seasons: list[int]) -> pd.DataFrame:
    st = _read("stats_player_season")
    if st.empty:
        return pd.DataFrame(columns=["player_uid"])
    st = st[st["season"].isin(seasons)].copy()
    st["player_uid"] = st["player_id"].map(resolver.canonical)
    st = st[st["player_uid"].notna()]

    st["team_carries"] = st.groupby(["season", "recent_team"])["carries"].transform("sum")
    st["carry_share"] = np.where(st["team_carries"] > 0,
                                 st["carries"] / st["team_carries"], np.nan)
    st["ppg"] = np.where(st["games"] > 0, st["fantasy_points_ppr"] / st["games"], np.nan)

    weights = _recency_weights(seasons)
    st["w"] = st["season"].map(weights) * st["games"].clip(lower=0)

    metrics = ["target_share", "air_yards_share", "wopr", "carry_share", "ppg",
               "racr", "receiving_epa", "rushing_epa"]
    metrics = [m for m in metrics if m in st.columns]

    def wavg(g: pd.DataFrame) -> pd.Series:
        w = g["w"].fillna(0)
        out = {}
        for m in metrics:
            v = pd.to_numeric(g[m], errors="coerce")
            mask = v.notna() & (w > 0)
            out[m] = np.average(v[mask], weights=w[mask]) if mask.any() else np.nan
        return pd.Series(out)

    agg = st.groupby("player_uid")[metrics + ["w"]].apply(wavg).reset_index()
    agg.columns = ["player_uid"] + [f"{m}_w3y" for m in metrics]

    # Prior-season standalone values, plus games played per season for durability.
    last = st[st["season"] == max(seasons)].copy()
    last["recent_team"] = last["recent_team"].map(norm_team)
    last_cols = last.groupby("player_uid").agg(
        games_ly=("games", "sum"),
        ppr_points_ly=("fantasy_points_ppr", "sum"),
        targets_ly=("targets", "sum"),
        carries_ly=("carries", "sum"),
        team_ly=("recent_team", "last"),
    ).reset_index()
    agg = agg.merge(last_cols, on="player_uid", how="outer")

    # Durability: games missed across the seasons the player was in the league.
    played = st.groupby(["player_uid", "season"])["games"].sum().reset_index()
    first_season = played.groupby("player_uid")["season"].min()
    played["eligible"] = played["player_uid"].map(first_season) <= played["season"]
    played["missed"] = np.where(played["eligible"],
                                (SEASON_GAMES - played["games"]).clip(lower=0), 0)
    missed = played.groupby("player_uid")["missed"].sum().rename("games_missed_l3y")
    agg = agg.merge(missed.reset_index(), on="player_uid", how="left")
    return agg


# ---------------------------------------------------------------------------
# Red zone / goal line share, and team pass tendency, from play-by-play
# ---------------------------------------------------------------------------
def pbp_features(resolver: IdResolver, seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pbp = _read("pbp")
    if pbp.empty:
        return pd.DataFrame(columns=["player_uid"]), pd.DataFrame(columns=["team"])

    last = max(seasons)
    p = pbp[pbp["season"] == last].copy()
    p["posteam"] = p["posteam"].map(norm_team)

    # --- player red-zone / goal-line touch share ---
    touches = []
    for col, kind in (("rusher_player_id", "rush"), ("receiver_player_id", "rec")):
        if col not in p.columns:
            continue
        d = p[p[col].notna()][["posteam", col, "yardline_100"]].copy()
        d = d.rename(columns={col: "pid"})
        d["kind"] = kind
        touches.append(d)
    if not touches:
        return pd.DataFrame(columns=["player_uid"]), pd.DataFrame(columns=["team"])
    tt = pd.concat(touches, ignore_index=True)
    tt["rz"] = (tt["yardline_100"] <= 20).astype(int)
    tt["gl"] = (tt["yardline_100"] <= 5).astype(int)

    team_tot = tt.groupby("posteam")[["rz", "gl"]].sum().rename(
        columns={"rz": "team_rz", "gl": "team_gl"})
    per = tt.groupby(["posteam", "pid"])[["rz", "gl"]].sum().reset_index()
    per = per.merge(team_tot, on="posteam", how="left")
    per["redzone_touch_share"] = per["rz"] / per["team_rz"].replace(0, np.nan)
    per["goalline_share"] = per["gl"] / per["team_gl"].replace(0, np.nan)
    per["player_uid"] = per["pid"].map(resolver.canonical)
    player_rz = (per[per["player_uid"].notna()]
                 .groupby("player_uid")[["redzone_touch_share", "goalline_share"]]
                 .max().reset_index())

    # --- team neutral-situation pass rate (PROE proxy) ---
    neutral = p[
        p["down"].isin([1, 2])
        & p["wp"].between(0.20, 0.80)
        & (p["half_seconds_remaining"] > 120)
        & p["play_type"].isin(["pass", "run"])
    ]
    team_pass = (neutral.assign(is_pass=neutral["play_type"].eq("pass"))
                 .groupby("posteam")["is_pass"].mean()
                 .rename("neutral_pass_rate").reset_index()
                 .rename(columns={"posteam": "team"}))
    team_pass["proe_z"] = _z(team_pass["neutral_pass_rate"])
    return player_rz, team_pass


# ---------------------------------------------------------------------------
# Snap trend: late-season usage weighted above the full-season average
# ---------------------------------------------------------------------------
def snap_features(resolver: IdResolver, season_prev: int) -> pd.DataFrame:
    snaps = _read("snap_counts")
    players = _read("players")
    if snaps.empty or players.empty:
        return pd.DataFrame(columns=["player_uid"])

    pfr = players[["pfr_id", "gsis_id"]].dropna()
    pfr_map = dict(zip(pfr["pfr_id"], pfr["gsis_id"]))

    s = snaps[(snaps["season"] == season_prev) & (snaps["game_type"] == "REG")].copy()
    s["player_uid"] = s["pfr_player_id"].map(pfr_map).map(resolver.canonical)
    s = s[s["player_uid"].notna()]
    s["offense_pct"] = pd.to_numeric(s["offense_pct"], errors="coerce")

    season_avg = s.groupby("player_uid")["offense_pct"].mean().rename("snap_pct_season")
    last4 = (s.sort_values("week").groupby("player_uid").tail(4)
             .groupby("player_uid")["offense_pct"].mean().rename("snap_pct_l4"))
    out = pd.concat([season_avg, last4], axis=1).reset_index()
    out["snap_pct_trend"] = out["snap_pct_l4"] - out["snap_pct_season"]
    return out


# ---------------------------------------------------------------------------
# Expected fantasy points (ffopportunity)
# ---------------------------------------------------------------------------
def expected_points_features(resolver: IdResolver, season_prev: int) -> pd.DataFrame:
    ep = _read("ffopportunity")
    if ep.empty:
        return pd.DataFrame(columns=["player_uid"])
    e = ep[ep["season"] == season_prev].copy()
    e["player_uid"] = e["player_id"].map(resolver.canonical)
    e = e[e["player_uid"].notna()]
    agg = e.groupby("player_uid").agg(
        exp_fantasy_points=("total_fantasy_points_exp", "sum"),
        actual_fantasy_points=("total_fantasy_points", "sum"),
        weeks_played=("week", "nunique"),
    ).reset_index()
    agg["exp_pts_delta"] = agg["actual_fantasy_points"] - agg["exp_fantasy_points"]
    agg["exp_pts_delta_pg"] = agg["exp_pts_delta"] / agg["weeks_played"].replace(0, np.nan)
    agg["exp_fantasy_points_pg"] = (
        agg["exp_fantasy_points"] / agg["weeks_played"].replace(0, np.nan))
    return agg


# ---------------------------------------------------------------------------
# NGS efficiency
# ---------------------------------------------------------------------------
def ngs_features(resolver: IdResolver, season_prev: int) -> pd.DataFrame:
    ngs = _read("ngs")
    if ngs.empty:
        return pd.DataFrame(columns=["player_uid"])
    n = ngs[(ngs["season"] == season_prev) & (ngs["week"] == 0)].copy()
    if n.empty:  # some releases only carry weekly rows
        n = ngs[ngs["season"] == season_prev].copy()
    n["player_uid"] = n["player_gsis_id"].map(resolver.canonical)
    n = n[n["player_uid"].notna()]

    wanted = {
        "avg_separation": "mean",
        "avg_intended_air_yards": "mean",
        "avg_yac_above_expectation": "mean",
        "catch_percentage": "mean",
        "rush_yards_over_expected_per_att": "mean",
        "rush_pct_over_expected": "mean",
        "avg_time_to_throw": "mean",
    }
    have = {k: v for k, v in wanted.items() if k in n.columns}
    if not have:
        return pd.DataFrame(columns=["player_uid"])
    return n.groupby("player_uid").agg(**{
        k: (k, v) for k, v in have.items()
    }).reset_index()


# ---------------------------------------------------------------------------
# Team-level context
# ---------------------------------------------------------------------------
def team_features(resolver: IdResolver, seasons: list[int],
                  team_pass: pd.DataFrame) -> pd.DataFrame:
    st = _read("stats_player_season")
    prev = max(seasons)

    teams = sorted({t for t in _read("rosters")["team"].map(norm_team) if t})
    out = pd.DataFrame({"team": teams})

    if not st.empty:
        s = st[st["season"] == prev].copy()
        s["recent_team"] = s["recent_team"].map(norm_team)

        # Target concentration: Herfindahl index of target share among the
        # team's pass catchers. High = funnels to one receiver.
        pc = s[s["position"].isin(["WR", "TE", "RB"])].copy()
        pc["ts"] = pd.to_numeric(pc["target_share"], errors="coerce").fillna(0)
        hhi = (pc.assign(sq=pc["ts"] ** 2)
                 .groupby("recent_team")["sq"].sum()
                 .rename("target_concentration").reset_index()
                 .rename(columns={"recent_team": "team"}))
        out = out.merge(hhi, on="team", how="left")

        # QB quality: passing EPA per attempt of the team's primary passer.
        qb = s[s["position"] == "QB"].copy()
        qb["attempts"] = pd.to_numeric(qb["attempts"], errors="coerce")
        qb = qb[qb["attempts"] >= 100]
        qb["epa_per_att"] = pd.to_numeric(qb["passing_epa"], errors="coerce") / qb["attempts"]
        primary = qb.sort_values("attempts").groupby("recent_team").tail(1)
        out = out.merge(
            primary[["recent_team", "epa_per_att"]]
            .rename(columns={"recent_team": "team", "epa_per_att": "qb_epa_per_att"}),
            on="team", how="left")

    # O-line proxy: team-level rushing yards over expected per attempt.
    # This conflates back talent with blocking, hence "proxy" - override it in
    # config/team_context.yaml when you have a better read.
    ngs = _read("ngs")
    if not ngs.empty and "rush_yards_over_expected_per_att" in ngs.columns:
        n = ngs[(ngs["season"] == prev) & (ngs["ngs_kind"] == "rushing")].copy()
        n["team"] = n["team_abbr"].map(norm_team)
        n["ryoe"] = pd.to_numeric(n["rush_yards_over_expected_per_att"], errors="coerce")
        n["att"] = pd.to_numeric(n.get("rush_attempts"), errors="coerce").fillna(1)
        grp = (n.dropna(subset=["ryoe"])
                 .groupby("team")
                 .apply(lambda g: np.average(g["ryoe"], weights=g["att"].clip(lower=1)),
                        include_groups=False)
                 .rename("oline_ryoe").reset_index())
        out = out.merge(grp, on="team", how="left")

    if not team_pass.empty:
        out = out.merge(team_pass, on="team", how="left")

    for col, zname in (("oline_ryoe", "oline_z"),
                       ("target_concentration", "target_concentration_z"),
                       ("qb_epa_per_att", "qb_epa_z")):
        if col in out.columns:
            out[zname] = _z(out[col])

    # QB tier 1-5 from EPA z-score (1 = best). Hand-editable downstream.
    if "qb_epa_z" in out.columns:
        out["qb_tier"] = pd.cut(
            out["qb_epa_z"], bins=[-np.inf, -1.0, -0.35, 0.35, 1.0, np.inf],
            labels=[5, 4, 3, 2, 1]).astype("float").fillna(3).astype(int)
    else:
        out["qb_tier"] = 3
    return out


# ---------------------------------------------------------------------------
# Schedule: bye weeks and positional strength of schedule
# ---------------------------------------------------------------------------
def schedule_features(season: int, seasons_prev: list[int]) -> pd.DataFrame:
    sched = _read("schedules")
    week = _read("stats_player_week")
    if sched.empty:
        return pd.DataFrame(columns=["team"])

    cur = sched[sched["season"] == season].copy()
    cur["home_team"] = cur["home_team"].map(norm_team)
    cur["away_team"] = cur["away_team"].map(norm_team)

    long = pd.concat([
        cur[["week", "home_team", "away_team"]].rename(
            columns={"home_team": "team", "away_team": "opp"}),
        cur[["week", "away_team", "home_team"]].rename(
            columns={"away_team": "team", "home_team": "opp"}),
    ], ignore_index=True)

    teams = sorted(long["team"].dropna().unique())
    max_week = int(cur["week"].max())
    byes = {}
    for t in teams:
        played = set(long[long["team"] == t]["week"])
        missing = [w for w in range(1, max_week + 1) if w not in played]
        byes[t] = missing[0] if missing else None
    bye_df = pd.DataFrame({"team": list(byes), "bye_week": list(byes.values())})

    # Defensive strength: PPR points allowed per game by position, last season.
    if week.empty:
        return bye_df
    prev = max(seasons_prev)
    w = week[(week["season"] == prev) & (week["season_type"] == "REG")].copy()
    w["opponent_team"] = w["opponent_team"].map(norm_team)
    w = w[w["position"].isin(["QB", "RB", "WR", "TE"])]
    allowed = (w.groupby(["opponent_team", "position", "week"])["fantasy_points_ppr"]
                 .sum().reset_index()
                 .groupby(["opponent_team", "position"])["fantasy_points_ppr"]
                 .mean().reset_index()
                 .rename(columns={"opponent_team": "opp",
                                  "fantasy_points_ppr": "pts_allowed_pg"}))
    allowed["def_z"] = allowed.groupby("position")["pts_allowed_pg"].transform(_z)

    # Weight early weeks far above late ones - week 16 matchups are not
    # knowable in August, week 1-6 roughly are.
    long["wk_weight"] = np.where(long["week"] <= 6, 1.0,
                          np.where(long["week"] <= 12, 0.6,
                          np.where(long["week"] <= 14, 0.3, 0.1)))
    merged = long.merge(allowed, on="opp", how="left")
    merged = merged.dropna(subset=["def_z"])
    sos = (merged.assign(wz=merged["def_z"] * merged["wk_weight"])
                 .groupby(["team", "position"])
                 .apply(lambda g: g["wz"].sum() / g["wk_weight"].sum(),
                        include_groups=False)
                 .rename("sos_z").reset_index())
    sos_wide = sos.pivot(index="team", columns="position", values="sos_z")
    sos_wide.columns = [f"sos_z_{c}" for c in sos_wide.columns]
    return bye_df.merge(sos_wide.reset_index(), on="team", how="left")


# ---------------------------------------------------------------------------
# Hand-editable team context overlay
# ---------------------------------------------------------------------------
def sync_team_context(team_df: pd.DataFrame) -> dict:
    """Write config/team_context.yaml with auto-derived values, preserving any
    field a human has already edited. Auto values live under `derived`, manual
    overrides under `override` - overrides always win."""
    path = CONFIG / "team_context.yaml"
    existing = {}
    if path.exists():
        with open(path) as fh:
            existing = yaml.safe_load(fh) or {}
    existing_teams = existing.get("teams", {}) or {}

    teams = {}
    for r in team_df.itertuples(index=False):
        t = r.team
        prior = existing_teams.get(t, {}) or {}
        derived = {
            "neutral_pass_rate": _round(getattr(r, "neutral_pass_rate", None)),
            "proe_z": _round(getattr(r, "proe_z", None)),
            "target_concentration": _round(getattr(r, "target_concentration", None)),
            "oline_ryoe": _round(getattr(r, "oline_ryoe", None)),
            "oline_z": _round(getattr(r, "oline_z", None)),
            "qb_epa_per_att": _round(getattr(r, "qb_epa_per_att", None)),
            "qb_tier": int(getattr(r, "qb_tier", 3) or 3),
        }
        teams[t] = {
            "derived": derived,
            # Anything you set here wins over `derived`. Leave null to defer.
            "override": prior.get("override", {
                "qb_tier": None, "oline_z": None, "proe_z": None,
                "target_concentration": None, "note": None,
            }),
        }

    doc = {
        "_readme": (
            "Auto-derived team context. `derived` is regenerated every time "
            "features.py runs; `override` is never touched - set a value there "
            "to overrule the derived number (e.g. a QB change the 2025 data "
            "cannot know about). qb_tier: 1 = best, 5 = worst."
        ),
        "teams": teams,
    }
    with open(path, "w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=True, default_flow_style=False)
    return doc


def _round(v, nd: int = 4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return round(float(v), nd)
    except (TypeError, ValueError):
        return None


def apply_overrides(team_df: pd.DataFrame, doc: dict) -> pd.DataFrame:
    out = team_df.copy()
    teams = doc.get("teams", {})
    for i, row in out.iterrows():
        ov = (teams.get(row["team"], {}) or {}).get("override", {}) or {}
        for key, val in ov.items():
            if key == "note" or val is None:
                continue
            if key in out.columns:
                out.at[i, key] = val
    return out


# ---------------------------------------------------------------------------
def build(season: int, history: int = 3) -> pd.DataFrame:
    ensure_dirs()
    resolver = IdResolver()
    seasons = [season - i for i in range(1, history + 1)]
    prev = max(seasons)

    print(f"=== Layer 2: features (season {season}, history {seasons}) ===")
    bio = bio_features(resolver, season)
    print(f"  bio               {len(bio):>5}")
    depth = depth_chart_features(resolver)
    print(f"  depth chart       {len(depth):>5}")
    vol = volume_features(resolver, seasons)
    print(f"  volume            {len(vol):>5}")
    rz, team_pass = pbp_features(resolver, seasons)
    print(f"  red zone          {len(rz):>5}")
    snaps = snap_features(resolver, prev)
    print(f"  snap trend        {len(snaps):>5}")
    exp = expected_points_features(resolver, prev)
    print(f"  expected points   {len(exp):>5}")
    ngs = ngs_features(resolver, prev)
    print(f"  ngs efficiency    {len(ngs):>5}")

    tf = team_features(resolver, seasons, team_pass)
    doc = sync_team_context(tf)
    tf = apply_overrides(tf, doc)
    sched = schedule_features(season, seasons)
    tf = tf.merge(sched, on="team", how="left")
    print(f"  team context      {len(tf):>5}  -> config/team_context.yaml")

    feat = bio
    for other in (depth, vol, rz, snaps, exp, ngs):
        if not other.empty:
            feat = feat.merge(other, on="player_uid", how="outer")
    feat = feat[feat["player_uid"].notna()].drop_duplicates("player_uid")

    # Current team: depth chart is fresher than the roster file.
    feat["team"] = (feat.get("depth_team")
                    .fillna(feat.get("roster_team"))
                    if "depth_team" in feat.columns else feat.get("roster_team"))
    feat["new_team_flag"] = (
        feat["team"].notna() & feat.get("team_ly").notna()
        & (feat["team"] != feat.get("team_ly"))
    ).astype(int)

    DATA_PROC.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(DATA_PROC / "player_features.parquet", index=False)
    tf.to_parquet(DATA_PROC / "team_features.parquet", index=False)
    print(f"  -> data/processed/player_features.parquet ({len(feat)} rows, "
          f"{len(feat.columns)} cols)")
    return feat


def main(argv=None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Layer 2 - build feature tables")
    ap.add_argument("--season", type=int, default=cfg["season"])
    ap.add_argument("--history", type=int, default=3)
    args = ap.parse_args(argv)
    build(args.season, args.history)
    return 0


if __name__ == "__main__":
    sys.exit(main())
