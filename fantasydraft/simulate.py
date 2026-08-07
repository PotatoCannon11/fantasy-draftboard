"""Monte Carlo draft simulator -> availability probabilities and VONA.

VBD answers "who is worth the most?". It cannot answer "should I take him now,
or will he still be there next round?" - and that second question is what
actually decides a draft. This module answers it by simulating the draft
thousands of times and recording, for each of your picks, who is still there.

Opponent model (config: simulation.opponents.model)
----------------------------------------------------
`adp_only`  - the original model. Each player's realised slot is drawn from
              Normal(ADP, ADP_stdev) and the draft order is the sort of those
              draws. Fast and unbiased in aggregate, but every simulated draft
              is internally incoherent: a team can end up with five RBs and no
              WR, because picks are independent.

`roster_aware` (default) - simulates pick by pick. Each team still values
              players by an ADP draw, but will not draft a position it has
              already filled to the brim, and gets a nudge toward positions
              where it still needs a starter. This makes each simulated draft a
              set of *valid rosters*, which is what fixes the availability
              curves - a receiver is not "taken" in a world where the drafting
              team would never have taken him.

              Crucially this is calibrated to data, not to folk wisdom. The
              nudge only breaks near-ties in ADP; it never drags a round-10
              player into round 2. `build()` prints a calibration report so you
              can confirm each player's *simulated* mean draft slot still lands
              on his real ADP. The model encodes "teams draft coherent rosters"
              (true, and checkable) - not "never draft a TE before round 5"
              (a myth the ADP data already refutes).

Both model all N teams, including yours: availability at your pick is measured
from the board state right before that pick.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from common import DATA_PROC, FANTASY_POS, ensure_dirs, load_config

DEFAULT_SIMS = 20000

# Sensible defaults if simulation.opponents is absent from the config.
DEFAULT_OPP = {
    "model": "roster_aware",
    "start_bonus": 10.0,     # ADP-picks of pull toward an unfilled starter slot
    "flex_bonus": 5.0,       # ... toward filling the FLEX with RB/WR/TE
    "caps": {"QB": 3, "RB": 7, "WR": 8, "TE": 3, "K": 2, "DEF": 2},
    "kdef_last_rounds": 3,   # K/DEF get no starter-nudge until this many rounds remain
}


def my_picks(n_teams: int, slot: int, rounds: int) -> list[int]:
    """Overall pick numbers for a snake draft from a given seat."""
    picks = []
    for r in range(1, rounds + 1):
        if r % 2 == 1:
            picks.append((r - 1) * n_teams + slot)
        else:
            picks.append(r * n_teams - slot + 1)
    return picks


def team_of_pick(n_teams: int, rounds: int) -> np.ndarray:
    """Which team (0-indexed) is on the clock at each overall pick, snake order."""
    seq = []
    for r in range(rounds):
        order = range(n_teams) if r % 2 == 0 else range(n_teams - 1, -1, -1)
        seq.extend(order)
    return np.array(seq, dtype=np.int64)


def _draft_inputs(board, cfg, n_teams, rounds):
    adp_col, sd_col = f"adp_{n_teams}", f"adp_sd_{n_teams}"
    if adp_col not in board.columns:
        raise SystemExit(f"{adp_col} missing - run the ADP ingest first")
    total = n_teams * rounds
    undrafted = total + 25
    adp = pd.to_numeric(board[adp_col], errors="coerce").fillna(undrafted).to_numpy(float)
    sd = pd.to_numeric(board.get(sd_col), errors="coerce").to_numpy(float)
    # FFC gives no stdev for thinly-drafted players; late picks scatter more.
    sd = np.where(np.isnan(sd) | (sd <= 0), np.maximum(4.0, adp * 0.28), sd)
    vbd = pd.to_numeric(board["vbd_score"], errors="coerce").fillna(-999).to_numpy(float)
    pos = board["position"].to_numpy()
    pos_idx = np.array([FANTASY_POS.index(p) if p in FANTASY_POS else 0 for p in pos])
    return adp, sd, vbd, pos, pos_idx


# ---------------------------------------------------------------------------
# Roster-aware opponent model
# ---------------------------------------------------------------------------
def _roster_aware(adp, sd, pos_idx, cfg, *, n_teams, rounds, picks, sims, rng):
    """Sequential, vectorised across sims. Returns availability snapshots taken
    just before each of your picks, plus per-player draft pick for calibration.
    """
    opp = {**DEFAULT_OPP, **(cfg.get("simulation", {}).get("opponents") or {})}
    roster = cfg["league"]["roster"]
    flex_elig_pos = cfg["league"]["flex_eligible"]

    npos = len(FANTASY_POS)
    caps = np.array([opp["caps"].get(p, 99) for p in FANTASY_POS], float)
    starter = np.array([roster.get(p, 0) for p in FANTASY_POS], float)  # excludes FLEX
    flex_mask = np.array([p in flex_elig_pos for p in FANTASY_POS])
    flex_total = float(starter[flex_mask].sum() + roster.get("FLEX", 0))
    kdef_idx = [FANTASY_POS.index(p) for p in ("K", "DEF") if p in FANTASY_POS]

    n = len(adp)
    draws = adp[None, :] + rng.standard_normal((sims, n)) * sd[None, :]
    avail = np.ones((sims, n), dtype=bool)
    counts = np.zeros((sims, n_teams, npos), dtype=np.int16)
    taken_at = np.zeros((sims, n), dtype=np.int32)      # 0 = never drafted

    teams = team_of_pick(n_teams, rounds)
    my_set = set(picks)
    avail_at: dict[int, np.ndarray] = {}
    rows = np.arange(sims)
    total = n_teams * rounds
    sb, fb = opp["start_bonus"], opp["flex_bonus"]

    for pick in range(1, total + 1):
        if pick in my_set:
            avail_at[pick] = avail.copy()
        t = teams[pick - 1]
        rnd = (pick - 1) // n_teams + 1
        cnt = counts[:, t, :]                            # (sims, npos)

        # Positional need -> a bonus (in ADP-pick units) that only breaks ties.
        need = np.where(cnt < starter, sb, 0.0)          # unfilled starter slot
        flex_have = cnt[:, flex_mask].sum(axis=1)        # (sims,)
        flex_open = (flex_have < flex_total)[:, None] & flex_mask[None, :]
        need = need + flex_open * fb
        # K/DEF are not "needed" until the endgame - ADP already sends them late,
        # and a starter-nudge would otherwise pull them up far too early.
        if rnd < rounds - opp["kdef_last_rounds"] + 1:
            need[:, kdef_idx] = 0.0

        cap_ok = cnt < caps                              # (sims, npos)
        eff = draws - need[:, pos_idx]                   # lower = more desirable
        eff[~avail] = np.inf
        eff[~cap_ok[:, pos_idx]] = np.inf

        choice = eff.argmin(axis=1)
        # Safety net: if a sim capped/exhausted every eligible position, fall
        # back to plain best-available so it still drafts a real player.
        stuck = ~np.isfinite(eff[rows, choice])
        if stuck.any():
            alt = np.where(avail, draws, np.inf)
            choice[stuck] = alt[stuck].argmin(axis=1)

        avail[rows, choice] = False
        taken_at[rows, choice] = pick
        counts[rows, t, pos_idx[choice]] += 1

    return avail_at, taken_at


def _adp_only(adp, sd, pos_idx, cfg, *, n_teams, rounds, picks, sims, rng):
    """Original independent-draw model, kept as a fast, roster-blind baseline."""
    n = len(adp)
    draws = adp[None, :] + rng.standard_normal((sims, n)) * sd[None, :]
    order = np.argsort(draws, axis=1)
    slots = np.empty_like(order)
    np.put_along_axis(slots, order,
                      np.arange(1, n + 1)[None, :].repeat(sims, 0), axis=1)
    avail_at = {k: (slots >= k) for k in picks}
    taken_at = slots.astype(np.int32)
    return avail_at, taken_at


# ---------------------------------------------------------------------------
def simulate(board: pd.DataFrame, cfg: dict, *, n_teams: int, slot: int,
             rounds: int, sims: int = DEFAULT_SIMS,
             seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    d = board.reset_index(drop=True)
    adp, sd, vbd, pos, pos_idx = _draft_inputs(d, cfg, n_teams, rounds)
    n = len(d)
    picks = my_picks(n_teams, slot, rounds)

    model = (cfg.get("simulation", {}).get("opponents") or {}).get(
        "model", DEFAULT_OPP["model"])
    engine = _roster_aware if model == "roster_aware" else _adp_only
    avail_at, taken_at = engine(adp, sd, pos_idx, cfg, n_teams=n_teams,
                                rounds=rounds, picks=picks, sims=sims, rng=rng)

    # --- availability + expected best-available VBD per position, per pick ---
    best_rows = []
    avail_prob = {}
    for k in picks:
        mask = avail_at[k]                               # (sims, n) bool
        avail_prob[k] = mask.mean(axis=0)
        for p in FANTASY_POS:
            pm = pos == p
            if not pm.any():
                continue
            v = np.where(mask[:, pm], vbd[pm][None, :], -np.inf)
            best = v.max(axis=1)
            best = np.where(np.isfinite(best), best, 0.0)
            best_rows.append({"pick": k, "position": p,
                              "exp_best_vbd": float(best.mean()),
                              "p10_best_vbd": float(np.percentile(best, 10)),
                              "p90_best_vbd": float(np.percentile(best, 90))})
    best_df = pd.DataFrame(best_rows)

    # --- VONA per player per pick ---
    nxt = {k: picks[i + 1] for i, k in enumerate(picks[:-1])}
    lut = best_df.set_index(["pick", "position"])["exp_best_vbd"].to_dict()
    out = []
    for k in picks:
        follow = nxt.get(k)
        for i in range(n):
            p = pos[i]
            base = lut.get((follow, p)) if follow else None
            out.append({
                "pick": k,
                "player_uid": d["player_uid"].iat[i],
                "player_name": d["player_name"].iat[i],
                "position": p,
                "vbd_score": float(vbd[i]) if vbd[i] > -999 else np.nan,
                "p_available_now": float(avail_prob[k][i]),
                "p_available_next": float(avail_prob[follow][i]) if follow else np.nan,
                "vona": (float(vbd[i]) - base) if (base is not None and vbd[i] > -999)
                        else np.nan,
            })
    vona_df = pd.DataFrame(out)

    # --- calibration: simulated mean draft slot vs real ADP ---
    drafted = taken_at > 0
    n_drafted = drafted.sum(axis=0)
    slot_sum = np.where(drafted, taken_at, 0).sum(axis=0)
    # Deep-bench players go undrafted in every sim; leave their mean slot NaN
    # rather than dividing by zero.
    mean_slot = np.divide(slot_sum, n_drafted, out=np.full(n, np.nan),
                          where=n_drafted > 0)
    calib = pd.DataFrame({
        "player_name": d["player_name"], "position": pos,
        "adp": adp, "sim_mean_slot": mean_slot,
        "sim_taken_pct": drafted.mean(axis=0),
    })
    return vona_df, best_df, calib


def _report_calibration(calib: pd.DataFrame, n_teams: int, rounds: int) -> None:
    window = n_teams * rounds
    core = calib[(calib["adp"] < window) & calib["sim_mean_slot"].notna()]
    if core.empty:
        return
    err = (core["sim_mean_slot"] - core["adp"]).abs()
    corr = core["adp"].corr(core["sim_mean_slot"])
    print(f"  calibration: sim slot vs ADP  corr={corr:.3f}  "
          f"median|err|={err.median():.1f} picks  (drafted-range players)")
    counts = (calib[calib["adp"] < window]
              .assign(taken=lambda x: x["sim_taken_pct"] > 0.5)
              .groupby("position")["taken"].sum().astype(int))
    obs = ", ".join(f"{p}{int(counts.get(p, 0))}" for p in FANTASY_POS)
    print(f"  simulated demand (>50% drafted): {obs}")


def build(cfg: dict, *, slot: int | None = None, sims: int | None = None,
          rounds: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    path = DATA_PROC / "draft_board.parquet"
    if not path.exists():
        raise SystemExit("draft_board.parquet missing - run board.py first")
    board = pd.read_parquet(path)
    board = board.sort_values("vbd_score", ascending=False).head(
        cfg["output"]["board_depth"]).reset_index(drop=True)

    n_teams = cfg["league"]["primary_team_count"]
    sim_cfg = cfg.get("simulation", {})
    slot = slot or sim_cfg.get("draft_slot", (n_teams + 1) // 2)
    sims = sims or sim_cfg.get("sims", DEFAULT_SIMS)
    rounds = rounds or sim_cfg.get("rounds",
                                   sum(cfg["league"]["roster"].values())
                                   + cfg["league"]["bench_slots"])
    model = (sim_cfg.get("opponents") or {}).get("model", DEFAULT_OPP["model"])

    print("=== Monte Carlo draft simulation ===")
    print(f"  {sims:,} drafts | {n_teams} teams | slot {slot} | {rounds} rounds "
          f"| opponents: {model}")
    vona, best, calib = simulate(board, cfg, n_teams=n_teams, slot=slot,
                                 rounds=rounds, sims=sims)
    _report_calibration(calib, n_teams, rounds)

    vona.to_parquet(DATA_PROC / "sim_vona.parquet", index=False)
    best.to_parquet(DATA_PROC / "sim_best_available.parquet", index=False)
    picks = sorted(vona["pick"].unique())
    print(f"  your picks: {', '.join(map(str, picks))}")
    print(f"  -> data/processed/sim_vona.parquet ({len(vona):,} rows)")
    return vona, best


def main(argv=None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Monte Carlo draft simulation")
    ap.add_argument("--slot", type=int, default=None, help="your draft position")
    ap.add_argument("--sims", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--model", choices=["roster_aware", "adp_only"], default=None)
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args(argv)
    if args.model:
        cfg.setdefault("simulation", {}).setdefault("opponents", {})["model"] = args.model
    vona, best = build(cfg, slot=args.slot, sims=args.sims, rounds=args.rounds)
    if args.show:
        first = sorted(vona["pick"].unique())[0]
        g = vona[vona["pick"] == first].nlargest(args.show, "vona")
        print("\nHighest VONA at your first pick:")
        print(g[["player_name", "position", "vbd_score", "p_available_next",
                 "vona"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
