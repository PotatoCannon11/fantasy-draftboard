"""Layer 3 - projection model.

Tier 1: wisdom-of-crowds blend. Each source is z-scored within position, the
z-scores are averaged with configurable weights, then mapped back onto the
pooled points scale for that position. This keeps a source that is simply
scaled differently (e.g. one that assumes 16 games) from dragging the blend.

Tier 2: context adjustment. Multipliers built from the Layer 2 features are
compounded onto the baseline, with the total swing clamped so no single player
can be moved further than `context.max_total_swing`.

Tier 3 (custom regression) is deliberately not built here - see README.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from common import (
    DATA_PROC,
    DATA_RAW,
    FANTASY_POS,
    ensure_dirs,
    load_config,
    norm_pos,
    norm_team,
)
from idmap import IdResolver, resolve_frame

SOURCES = {
    "sleeper": {"file": "proj_sleeper", "id_col": "sleeper_id", "kind": "sleeper"},
    "espn": {"file": "proj_espn", "id_col": "espn_id", "kind": "espn"},
    "fantasysharks": {"file": "proj_fantasysharks", "id_col": None, "kind": "name"},
}


def _read_raw(name: str) -> pd.DataFrame:
    path = DATA_RAW / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


# ---------------------------------------------------------------------------
# Tier 1 - blend
# ---------------------------------------------------------------------------
def load_sources(resolver: IdResolver) -> pd.DataFrame:
    """Long frame: one row per (player, source)."""
    frames = []
    for name, spec in SOURCES.items():
        raw = _read_raw(spec["file"])
        if raw.empty:
            print(f"  [warn] projection source '{name}' missing")
            continue
        kw = {}
        if spec["kind"] == "sleeper":
            kw["sleeper_col"] = "sleeper_id"
        elif spec["kind"] == "espn":
            kw["espn_col"] = "espn_id"
        d = resolve_frame(raw, resolver, **kw)
        d["source"] = name
        d["position"] = d["position"].map(norm_pos)
        d["team"] = d["team"].map(norm_team)
        keep = ["player_uid", "player_name", "position", "team", "proj_points", "source"]
        frames.append(d[[c for c in keep if c in d.columns]])
        print(f"  {name:15s} {len(d):>5} players")
    if not frames:
        raise SystemExit("no projection sources available - run ingest first")
    return pd.concat(frames, ignore_index=True)


_RESOLVER_NAMES: dict[str, str] = {}


def _best_name(s: pd.Series) -> str:
    """FantasySharks publishes names as "Cook, James"; prefer a source that
    uses natural order, then the most common spelling."""
    vals = [v for v in s.dropna() if str(v).strip()]
    natural = [v for v in vals if "," not in str(v)]
    pool = natural or vals
    return pd.Series(pool).mode().iat[0] if pool else ""


def _display_name(uid: str, fallback: str, position: str, team: str) -> str:
    if position == "DEF":
        return f"{team} DEF" if team else str(fallback)
    canon = _RESOLVER_NAMES.get(uid)
    if canon:
        return canon
    name = str(fallback)
    if "," in name:  # last-resort flip
        last, _, first = name.partition(",")
        name = f"{first.strip()} {last.strip()}".strip()
    return name


def blend(long: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    pcfg = cfg["projections"]
    weights = pcfg["sources"]

    long = long[long["position"].isin(FANTASY_POS)].copy()
    long = long[long["proj_points"].notna() & (long["proj_points"] > 0)]
    long["w"] = long["source"].map(weights).fillna(0.0)
    long = long[long["w"] > 0]

    # Resolve the canonical name/position/team by majority across sources.
    ident = (long.groupby("player_uid")
                 .agg(player_name=("player_name", _best_name),
                      position=("position", lambda s: s.mode().iat[0]),
                      team=("team", lambda s: s.mode().iat[0] if s.notna().any() else None))
                 .reset_index())
    ident["player_name"] = [
        _display_name(uid, nm, pos, tm) for uid, nm, pos, tm in
        zip(ident["player_uid"], ident["player_name"], ident["position"], ident["team"])
    ]
    long = long.drop(columns=["player_name", "position", "team"]).merge(
        ident, on="player_uid", how="left")

    if pcfg.get("zscore_within_position", True):
        grp = long.groupby(["source", "position"])["proj_points"]
        long["z"] = ((long["proj_points"] - grp.transform("mean"))
                     / grp.transform("std").replace(0, np.nan)).fillna(0.0)
    else:
        long["z"] = long["proj_points"]

    agg = long.groupby(["player_uid", "position"]).apply(
        lambda g: pd.Series({
            "blend_z": np.average(g["z"], weights=g["w"]),
            "n_sources": g["source"].nunique(),
            "src_mean_points": np.average(g["proj_points"], weights=g["w"]),
            "src_spread": g["proj_points"].std(ddof=0),
            "src_min": g["proj_points"].min(),
            "src_max": g["proj_points"].max(),
        }), include_groups=False).reset_index()

    # Map the blended z back onto the pooled points scale for the position, so
    # the output is in real fantasy points rather than standard deviations.
    pooled = long.groupby("position")["proj_points"].agg(["mean", "std"])
    agg = agg.merge(pooled, left_on="position", right_index=True, how="left")
    agg["baseline_projection"] = agg["blend_z"] * agg["std"] + agg["mean"]
    agg = agg.drop(columns=["mean", "std"])
    # Fantasy points are bounded below by zero, but a linear z-to-points map is
    # not, so the deep tail can land slightly negative. Those players are far
    # outside the draftable range; floor them rather than let a negative
    # projection invert the variance band.
    agg["baseline_projection"] = agg["baseline_projection"].clip(lower=0.0)

    # Disagreement between sources, normalized - feeds the variance model.
    agg["src_cv"] = (agg["src_spread"] / agg["src_mean_points"].replace(0, np.nan)).fillna(0)

    agg = agg.merge(ident.drop(columns=["position"]), on="player_uid", how="left")
    agg["single_source"] = (agg["n_sources"] < cfg["projections"]["min_sources"]).astype(int)
    return agg


# ---------------------------------------------------------------------------
# Tier 2 - context multipliers
# ---------------------------------------------------------------------------
def _clip_mult(x, lo, hi):
    return float(np.clip(x, lo, hi))


def context_multipliers(df: pd.DataFrame, team: pd.DataFrame,
                        cfg: dict) -> pd.DataFrame:
    c = cfg["context"]
    out = df.copy()
    tm = team.set_index("team")

    def team_val(t, col, default=0.0):
        if t in tm.index and col in tm.columns:
            v = tm.at[t, col]
            if pd.notna(v):
                return float(v)
        return default

    pass_catcher = out["position"].isin(["WR", "TE"])
    rusher = out["position"].eq("RB")

    # -- PROE: pass-heavy offenses lift receivers, mildly suppress rushers ----
    m_proe = pd.Series(1.0, index=out.index)
    if c["proe"]["enabled"]:
        proe = out["team"].map(lambda t: team_val(t, "proe_z"))
        m_proe = np.where(
            pass_catcher, 1 + c["proe"]["pass_catcher_scale"] * proe * 0.05,
            np.where(rusher, 1 + c["proe"]["rusher_scale"] * proe * 0.05, 1.0))
        m_proe = pd.Series(m_proe, index=out.index)
    out["m_proe"] = m_proe

    # -- expected-points regression -----------------------------------------
    m_exp = pd.Series(1.0, index=out.index)
    if c["exp_pts_regression"]["enabled"] and "exp_pts_delta_pg" in out.columns:
        d = pd.to_numeric(out["exp_pts_delta_pg"], errors="coerce")
        z = d.groupby(out["position"]).transform(
            lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) else s * 0)
        scale = c["exp_pts_regression"]["scale"]
        cap = c["exp_pts_regression"]["cap"]
        m_exp = (1 - (z.fillna(0) * scale)).clip(1 - cap, 1 + cap)
    out["m_exp_regression"] = m_exp

    # -- target competition --------------------------------------------------
    m_comp = pd.Series(1.0, index=out.index)
    if c["target_competition"]["enabled"]:
        conc = out["team"].map(lambda t: team_val(t, "target_concentration_z"))
        scale = c["target_competition"]["scale"]
        # A concentrated target tree helps that team's WR1/TE1 and hurts the
        # rest of the room; a spread offense does the reverse.
        is_alpha = out.get("depth_rank", pd.Series(np.nan, index=out.index)).le(1)
        m_comp = pd.Series(
            np.where(pass_catcher & is_alpha, 1 + scale * conc * 0.5,
                     np.where(pass_catcher, 1 - scale * conc * 0.5, 1.0)),
            index=out.index)
    out["m_target_competition"] = m_comp

    # -- depth chart ---------------------------------------------------------
    m_depth = pd.Series(1.0, index=out.index)
    if c["depth_chart"]["enabled"] and "depth_rank" in out.columns:
        table = {int(k): float(v) for k, v in c["depth_chart"]["rank_multiplier"].items()}
        default = c["depth_chart"]["default"]
        rank = pd.to_numeric(out["depth_rank"], errors="coerce")
        m_depth = rank.map(lambda r: table.get(int(r), default) if pd.notna(r) else 1.0)
        # Kickers and defenses have no meaningful offensive depth rank.
        m_depth = m_depth.where(~out["position"].isin(["K", "DEF"]), 1.0)
    out["m_depth_chart"] = m_depth

    # -- durability ----------------------------------------------------------
    m_inj = pd.Series(1.0, index=out.index)
    if c["injury"]["enabled"] and "games_missed_l3y" in out.columns:
        gm = pd.to_numeric(out["games_missed_l3y"], errors="coerce").fillna(0)
        m_inj = (1 - gm * c["injury"]["per_game_missed"]).clip(
            lower=1 - c["injury"]["max_discount"], upper=1.0)
    out["m_injury"] = m_inj

    # -- age curve -----------------------------------------------------------
    m_age = pd.Series(1.0, index=out.index)
    if c["age"]["enabled"] and "age" in out.columns:
        curves = c["age"]["curves"]

        def age_mult(row):
            cur = curves.get(row["position"])
            age = row.get("age")
            if not cur or pd.isna(age):
                return 1.0
            over = max(0.0, float(age) - cur["peak_age"])
            return _clip_mult(1 - over * cur["per_year"],
                              1 - cur["max_discount"], 1.0)

        m_age = out.apply(age_mult, axis=1)
    out["m_age"] = m_age

    # -- QB tier applied to that team's pass catchers -------------------------
    m_qb = pd.Series(1.0, index=out.index)
    if c["qb_tier"]["enabled"]:
        table = {int(k): float(v) for k, v in c["qb_tier"]["multipliers"].items()}
        tiers = out["team"].map(lambda t: int(team_val(t, "qb_tier", 3)))
        m_qb = pd.Series(
            np.where(pass_catcher, tiers.map(lambda t: table.get(t, 1.0)), 1.0),
            index=out.index)
    out["m_qb_tier"] = m_qb

    # -- O-line applied to rushers -------------------------------------------
    m_ol = pd.Series(1.0, index=out.index)
    if c["oline"]["enabled"]:
        ol = out["team"].map(lambda t: team_val(t, "oline_z"))
        m_ol = pd.Series(np.where(rusher, 1 + c["oline"]["scale"] * ol, 1.0),
                         index=out.index)
    out["m_oline"] = m_ol

    mult_cols = [c_ for c_ in out.columns if c_.startswith("m_")]
    total = out[mult_cols].prod(axis=1)
    swing = c["max_total_swing"]
    out["context_multiplier"] = total.clip(1 - swing, 1 + swing)
    return out


def apply_shrinkage(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Pull each projection a little toward the positional mean.

    Held-out testing on 2014-2025 found k=0.30 optimal when shrinking last
    season's PPG. That figure does not carry over to a consensus projection,
    which has already been regressed by each source, so `k_base` is set far
    lower. Low-confidence players - those below `min_sources` - get shrunk
    harder, since a lone source is a guess rather than a consensus.
    """
    scfg = cfg["projections"].get("shrinkage", {})
    out = df.copy()
    if not scfg.get("enabled", False):
        out["final_projection"] = out["adjusted_projection"]
        out["shrinkage_k"] = 0.0
        return out

    k_base = float(scfg.get("k_base", 0.0))
    k_single = float(scfg.get("k_single_source", k_base))
    single = pd.to_numeric(out.get("single_source"), errors="coerce").fillna(0) > 0
    k = pd.Series(np.where(single, k_single, k_base), index=out.index)

    # Shrink toward the mean of *startable* players at the position rather than
    # the whole pool, which is dominated by deep-bench zeros.
    target = out.groupby("position")["adjusted_projection"].transform(
        lambda s: s.nlargest(max(3, int(len(s) * 0.4))).mean())

    out["shrinkage_k"] = k
    out["final_projection"] = (1 - k) * out["adjusted_projection"] + k * target
    return out


def project(cfg: dict) -> pd.DataFrame:
    ensure_dirs()
    print("=== Layer 3: projections ===")
    resolver = IdResolver()
    # nflverse spellings are the cleanest, so use them for display where known.
    _RESOLVER_NAMES.update({
        uid: rec["player_name"] for uid, rec in resolver.info.items()
        if rec.get("player_name")
    })
    long = load_sources(resolver)
    base = blend(long, cfg)
    print(f"  blended           {len(base):>5} players "
          f"({(base['n_sources'] >= 2).sum()} with 2+ sources)")

    feat_path = DATA_PROC / "player_features.parquet"
    team_path = DATA_PROC / "team_features.parquet"
    feat = pd.read_parquet(feat_path) if feat_path.exists() else pd.DataFrame()
    team = pd.read_parquet(team_path) if team_path.exists() else pd.DataFrame()
    if feat.empty:
        raise SystemExit("player_features.parquet missing - run features.py first")

    drop = {"player_name", "position"}
    fcols = [c for c in feat.columns if c not in drop]
    df = base.merge(feat[fcols], on="player_uid", how="left", suffixes=("", "_feat"))
    # Projection sources know a player's current team better than a roster
    # snapshot does; fall back to the feature table only when they disagree.
    df["team"] = df["team"].fillna(df.get("team_feat"))

    df = context_multipliers(df, team, cfg)
    df["adjusted_projection"] = df["baseline_projection"] * df["context_multiplier"]
    df = apply_shrinkage(df, cfg)

    # Attach bye week and SOS for the output sheet.
    if not team.empty:
        keep = ["team", "bye_week"] + [c for c in team.columns if c.startswith("sos_z_")]
        df = df.merge(team[keep], on="team", how="left")
        df["sos_z"] = df.apply(
            lambda r: r.get(f"sos_z_{r['position']}", np.nan), axis=1)

    df = df.sort_values("final_projection", ascending=False).reset_index(drop=True)
    df.to_parquet(DATA_PROC / "projections.parquet", index=False)
    print(f"  -> data/processed/projections.parquet ({len(df)} rows)")
    return df


def main(argv=None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Layer 3 - blend and adjust projections")
    ap.add_argument("--show", type=int, default=0, help="print top N")
    args = ap.parse_args(argv)
    df = project(cfg)
    if args.show:
        cols = ["player_name", "position", "team", "n_sources",
                "baseline_projection", "context_multiplier", "final_projection"]
        print(df[cols].head(args.show).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
