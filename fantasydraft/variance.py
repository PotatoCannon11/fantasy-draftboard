"""Layer 4 - uncertainty / variance modelling.

Each player gets a coefficient of variation built from a positional base and
widened by the things that actually make a season hard to forecast: an
ambiguous role, no NFL track record, a new team, injury history, and
disagreement between the projection sources.

Output: projection_low / projection_mid / projection_high plus the raw sigma,
which Layer 5 uses to sort Early / Mid / Late inside each tier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_variance(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    vcfg = cfg["variance"]
    base = vcfg["base_cv"]
    widen = vcfg["widen"]
    out = df.copy()

    cv = out["position"].map(base).astype(float).fillna(0.25)

    # Role clarity: anyone who is not the clear starter at their position on
    # the current depth chart carries extra downside.
    depth = pd.to_numeric(out.get("depth_rank"), errors="coerce")
    unclear = out["position"].isin(["RB", "WR", "TE", "QB"]) & (
        depth.isna() | (depth > 1))
    cv = cv + unclear.astype(float) * widen["unclear_role"]

    # No usable NFL sample.
    rookie = pd.to_numeric(out.get("rookie_flag"), errors="coerce").fillna(0) > 0
    cv = cv + rookie.astype(float) * widen["rookie"]

    new_team = pd.to_numeric(out.get("new_team_flag"), errors="coerce").fillna(0) > 0
    cv = cv + new_team.astype(float) * widen["new_team"]

    gm = pd.to_numeric(out.get("games_missed_l3y"), errors="coerce").fillna(0)
    cv = cv + (gm >= 8).astype(float) * widen["injury_history"]

    # Cross-source disagreement, normalized against the typical spread for the
    # position so a noisy position is not penalised twice.
    src_cv = pd.to_numeric(out.get("src_cv"), errors="coerce").fillna(0)
    rel = src_cv / src_cv.groupby(out["position"]).transform(
        lambda s: s.mean() if s.mean() else 1.0)
    cv = cv + (rel.clip(0, 3) - 1).clip(lower=0) * widen["low_source_agreement"]

    # A single-source projection is a guess, not a consensus.
    single = pd.to_numeric(out.get("single_source"), errors="coerce").fillna(0) > 0
    cv = cv + single.astype(float) * widen["low_source_agreement"]

    out["projection_cv"] = cv
    out["projection_sigma"] = out["final_projection"] * cv
    z = vcfg["band_z"]
    out["projection_mid"] = out["final_projection"]
    out["projection_low"] = (out["final_projection"] - z * out["projection_sigma"]).clip(lower=0)
    out["projection_high"] = out["final_projection"] + z * out["projection_sigma"]

    # Floor/ceiling relative to the position, for the Early/Mid/Late labels.
    out["floor_z"] = out.groupby("position")["projection_low"].transform(_z)
    out["risk_z"] = out.groupby("position")["projection_cv"].transform(_z)
    return out


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if not sd or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return ((s - s.mean()) / sd).fillna(0.0)
