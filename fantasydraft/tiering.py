"""Layer 5 - value conversion (VBD) and tiering.

Replacement level is derived from the actual league settings rather than a
fixed rule of thumb: starters at the position, plus that position's expected
share of the single FLEX slot, plus a streaming allowance for how many extra
bodies teams really roster. VBD is computed for every configured league size
so the board shows how sensitive a player is to that setting.

Tiers come from score-gap analysis: a new tier starts where the drop to the
next player is large relative to the local spread of scores, which finds the
real cliffs instead of slicing every N players.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import FANTASY_POS


# ---------------------------------------------------------------------------
# VBD
# ---------------------------------------------------------------------------
def replacement_ranks_from_adp(cfg: dict, n_teams: int,
                               adp: pd.DataFrame) -> dict[str, float] | None:
    """Set replacement level from what real drafters actually do.

    Counts how many players at each position are taken inside the first
    ``n_teams * roster_size`` picks of the FFC ADP sample. This is strictly
    better than a theoretical formula: it captures bench hoarding (people
    roster far more WRs than the starter math implies) and position-specific
    draft-day behaviour, and it re-derives itself each year.
    """
    if adp is None or adp.empty:
        return None
    g = adp[adp["teams"] == n_teams]
    if g.empty or "adp" not in g.columns:
        return None

    picks = n_teams * cfg["vbd"].get("roster_size", 15)
    taken = g.sort_values("adp").head(picks)
    counts = taken["position"].value_counts()
    floors = cfg["vbd"].get("min_rank", {})

    ranks = {}
    for pos in FANTASY_POS:
        n = int(counts.get(pos, 0))
        # The replacement player is the first one NOT drafted.
        ranks[pos] = float(max(n + 1, floors.get(pos, 1)))
    return ranks


def replacement_ranks(cfg: dict, n_teams: int) -> dict[str, float]:
    """How many players at each position come off the board before the pool is
    exhausted - i.e. the rank of the replacement-level player."""
    roster = cfg["league"]["roster"]
    flex_split = cfg["vbd"]["flex_split"]
    stream = cfg["vbd"]["streaming_allowance"]
    flex_slots = roster.get("FLEX", 0)

    ranks = {}
    for pos in FANTASY_POS:
        starters = roster.get(pos, 0)
        flex_share = flex_split.get(pos, 0.0) * flex_slots
        per_team = starters + flex_share + stream.get(pos, 0.0)
        ranks[pos] = max(1.0, per_team * n_teams)
    return ranks


def resolve_replacement(cfg: dict, n_teams: int,
                        adp: pd.DataFrame | None = None) -> tuple[dict, str]:
    """Replacement ranks plus which method produced them."""
    if cfg["vbd"].get("method", "static") == "adp_demand":
        ranks = replacement_ranks_from_adp(cfg, n_teams, adp)
        if ranks:
            return ranks, "adp_demand"
    return replacement_ranks(cfg, n_teams), "static"


def add_vbd(df: pd.DataFrame, cfg: dict,
            adp: pd.DataFrame | None = None) -> pd.DataFrame:
    out = df.copy()
    for n_teams in cfg["league"]["team_counts"]:
        ranks, _method = resolve_replacement(cfg, n_teams, adp)
        col = f"vbd_{n_teams}"
        out[col] = np.nan
        for pos, rank in ranks.items():
            mask = out["position"] == pos
            pool = out.loc[mask, "final_projection"].sort_values(ascending=False)
            if pool.empty:
                continue
            # Interpolate between the two players bracketing the fractional
            # replacement rank, so a 0.45 FLEX share is not rounded away.
            idx = min(max(rank - 1, 0), len(pool) - 1)
            lo, hi = int(np.floor(idx)), int(np.ceil(idx))
            frac = idx - lo
            baseline = pool.iloc[lo] * (1 - frac) + pool.iloc[hi] * frac
            out.loc[mask, col] = out.loc[mask, "final_projection"] - baseline
            out.loc[mask, f"replacement_{n_teams}"] = baseline

    primary = cfg["league"]["primary_team_count"]
    out["vbd_score"] = out[f"vbd_{primary}"]
    out["overall_rank"] = out["vbd_score"].rank(ascending=False, method="min")
    out["pos_rank"] = out.groupby("position")["vbd_score"].rank(
        ascending=False, method="min")

    # FLEX overlay: one combined RB/WR/TE scale so flex decisions are visible.
    flex_pos = cfg["league"]["flex_eligible"]
    fmask = out["position"].isin(flex_pos)
    out["flex_rank"] = np.nan
    out.loc[fmask, "flex_rank"] = out.loc[fmask, "vbd_score"].rank(
        ascending=False, method="min")
    return out


# ---------------------------------------------------------------------------
# Tiering
# ---------------------------------------------------------------------------
def _tiers_by_gap(scores: pd.Series, cfg: dict) -> np.ndarray:
    """Walk down the sorted scores and open a new tier wherever the gap to the
    next player is large relative to the local rolling spread."""
    tcfg = cfg["tiering"]
    s = scores.to_numpy(dtype=float)
    n = len(s)
    if n == 0:
        return np.array([], dtype=int)
    if n <= tcfg["min_tier_size"]:
        return np.ones(n, dtype=int)

    gaps = np.diff(s) * -1  # descending scores -> positive drops
    window = max(3, int(tcfg["rolling_window"]))

    # Compare each gap against the spread of SCORES in its neighbourhood, not
    # the spread of gaps. Gap-to-gap variation is tiny and roughly uniform, so
    # scoring against it fires a break almost everywhere; the local score
    # spread is what makes a genuine cliff stand out.
    local_sd = pd.Series(s).rolling(window, min_periods=3, center=True).std()
    fallback = np.nanstd(s) or 1.0
    local_sd = local_sd.bfill().ffill().fillna(fallback).to_numpy()
    local_sd = np.where((local_sd <= 0) | np.isnan(local_sd), fallback, local_sd)

    threshold = tcfg["gap_threshold"] * local_sd[:-1]
    tiers = np.ones(n, dtype=int)
    current = 1
    size = 1
    for i in range(1, n):
        is_break = gaps[i - 1] > threshold[i - 1]
        if is_break and size >= tcfg["min_tier_size"] and current < tcfg["max_tiers_per_position"]:
            current += 1
            size = 0
        tiers[i] = current
        size += 1
    return tiers


def _tiers_by_kmeans(scores: pd.Series, position: str, cfg: dict) -> np.ndarray:
    from sklearn.cluster import KMeans

    k = cfg["tiering"]["kmeans_k"].get(position, 6)
    x = scores.to_numpy(dtype=float).reshape(-1, 1)
    k = min(k, len(x))
    if k < 2:
        return np.ones(len(x), dtype=int)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(x)
    # Relabel clusters so tier 1 is the highest-scoring group.
    order = np.argsort(-km.cluster_centers_.ravel())
    remap = {old: new + 1 for new, old in enumerate(order)}
    return np.array([remap[c] for c in km.labels_], dtype=int)


def add_tiers(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    method = cfg["tiering"]["method"]
    out = df.copy()
    out["tier"] = np.nan

    for pos, grp in out.groupby("position"):
        g = grp.sort_values("vbd_score", ascending=False)
        if method == "kmeans":
            tiers = _tiers_by_kmeans(g["vbd_score"], pos, cfg)
        else:
            tiers = _tiers_by_gap(g["vbd_score"], cfg)
        out.loc[g.index, "tier"] = tiers
    out["tier"] = out["tier"].astype(int)

    # Cross-check: how far the two methods disagree is a useful confidence hint.
    if method == "gap":
        alt = pd.Series(index=out.index, dtype=float)
        for pos, grp in out.groupby("position"):
            g = grp.sort_values("vbd_score", ascending=False)
            try:
                alt.loc[g.index] = _tiers_by_kmeans(g["vbd_score"], pos, cfg)
            except Exception:  # noqa: BLE001
                alt.loc[g.index] = np.nan
        out["tier_kmeans"] = alt

    # Early / Mid / Late inside each tier: highest floor first, highest
    # variance last. This is the draft-order hint within a tier of equals.
    out["sub_tier_label"] = ""
    for (_pos, _tier), grp in out.groupby(["position", "tier"]):
        n = len(grp)
        if n == 1:
            out.loc[grp.index, "sub_tier_label"] = "Mid"
            continue
        score = grp["floor_z"].fillna(0) - grp["risk_z"].fillna(0)
        # 0-indexed position within the tier, safest first.
        order = score.rank(ascending=False, method="first").astype(int) - 1
        if n == 2:
            # A pair splits into safest / riskiest; "Mid" would be meaningless.
            labels = order.map({0: "Early", 1: "Late"})
        else:
            labels = order.map(
                lambda i: ["Early", "Mid", "Late"][min(int(3 * i // n), 2)])
        out.loc[grp.index, "sub_tier_label"] = labels

    out["tier_label"] = out["position"] + out["tier"].astype(str)
    return out
