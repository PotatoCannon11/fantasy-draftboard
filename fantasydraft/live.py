"""The live draft engine, in Python.

This is a port of the logic that used to live only as worksheet formulas on the
Pick Assistant / LiveCalc tabs. Same maths, same config knobs - but ordinary
code, so it can be unit-tested directly instead of needing a headless
LibreOffice recalculation to prove it evaluates at all.

`test_live.py` asserts parity with the workbook's own recalculated numbers, so
the two surfaces cannot silently drift apart while the xlsx remains the backup.

The one thing this can do that the spreadsheet could not: the tracker only ever
knew a *set* of marked players, so it could not see pick ORDER. Here the draft
is a list, which makes a real windowed run detector possible (`runs()`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from common import FANTASY_POS

# How deep into each position's queue the availability model looks. Matches the
# `depth` argument of the workbook's LiveCalc sheet.
DEPTH = 15
UNDRAFTED_ADP = 999.0
DEFAULT_SD = 10.0


def my_picks(n_teams: int, slot: int, rounds: int) -> list[int]:
    """Overall pick numbers for a snake draft from a given seat."""
    out = []
    for r in range(1, rounds + 1):
        out.append((r - 1) * n_teams + slot if r % 2 == 1
                   else r * n_teams - slot + 1)
    return out


@dataclass
class Pick:
    idx: int          # row index into the board frame
    mine: bool = False


@dataclass
class DraftState:
    """Everything that changes during a draft. `order` is the source of truth;
    the sets are derived so undo is just a pop."""
    order: list[Pick] = field(default_factory=list)
    override: int | None = None

    @property
    def drafted(self) -> set[int]:
        return {p.idx for p in self.order}

    @property
    def mine(self) -> list[int]:
        return [p.idx for p in self.order if p.mine]

    def take(self, idx: int, mine: bool = False) -> None:
        if idx not in self.drafted:
            self.order.append(Pick(idx, mine))

    def undo(self) -> Pick | None:
        return self.order.pop() if self.order else None

    def unmark(self, idx: int) -> None:
        self.order = [p for p in self.order if p.idx != idx]


class LiveBoard:
    """Draft-state-aware view of the board."""

    def __init__(self, board: pd.DataFrame, cfg: dict):
        self.cfg = cfg
        self.board = board.sort_values("vbd_score", ascending=False)\
                          .reset_index(drop=True)
        n = len(self.board)
        self.n_teams = int(cfg["league"]["primary_team_count"])
        self.rounds = int(cfg["simulation"]["rounds"])
        self.slot = int(cfg["simulation"]["draft_slot"])
        ocfg = cfg.get("output", {})
        self.useful = {**{p: 99 for p in FANTASY_POS},
                       **(ocfg.get("useful_max") or {})}
        self.run_window = int(ocfg.get("positional_run_window", 8))

        self.pos = self.board["position"].to_numpy()
        self.names = self.board["player_name"].to_numpy()
        # The pipeline precomputes VBD and ADP for every size in
        # league.team_counts, so changing team count at the draft table picks a
        # different precomputed column instead of needing a rebuild. A size that
        # was never precomputed falls back to the built-in one - `stale_for()`
        # reports that so the UI can say so rather than quietly lying.
        self.vbd_col = self._pick_col("vbd", "vbd_score")
        self.adp_col = self._pick_col("adp", f"adp_{ocfg.get('adp_teams', 10)}")
        self.sd_col = self._pick_col("adp_sd",
                                     f"adp_sd_{ocfg.get('adp_teams', 10)}")
        self.vbd = pd.to_numeric(self.board[self.vbd_col],
                                 errors="coerce").fillna(0.0).to_numpy(float)
        self.tier = pd.to_numeric(self.board.get("tier"),
                                  errors="coerce").fillna(99).to_numpy(int)
        adp = pd.to_numeric(self.board.get(self.adp_col), errors="coerce")
        self.adp = adp.fillna(UNDRAFTED_ADP).to_numpy(float)
        sd = pd.to_numeric(self.board.get(self.sd_col),
                           errors="coerce").to_numpy(float)
        # Same fallback the simulator and the tracker's hidden column use.
        self.sd = np.where(np.isnan(sd) | (sd <= 0),
                           np.maximum(4.0, self.adp * 0.28), sd)
        self.sd = np.where(self.sd <= 0, DEFAULT_SD, self.sd)

        self.my_pick_nos = my_picks(self.n_teams, self.slot, self.rounds)
        self._pos_rows = {p: np.where(self.pos == p)[0] for p in FANTASY_POS}
        assert n == len(self.vbd)

    def _pick_col(self, prefix: str, fallback: str) -> str:
        col = f"{prefix}_{self.n_teams}"
        return col if col in self.board.columns else fallback

    def stale_for(self) -> list[str]:
        """Which numbers are NOT valid for the current team count, because the
        pipeline never precomputed them at this size. Anything listed here needs
        a `run_pipeline.py --stage board` to be exact."""
        bad = []
        if f"vbd_{self.n_teams}" not in self.board.columns:
            bad.append(f"VBD (using {self.vbd_col})")
        if f"adp_{self.n_teams}" not in self.board.columns:
            bad.append(f"ADP (using {self.adp_col})")
        # Tiers are cut once, by the pipeline, at whatever primary size was on
        # disk at build time. Comparing against the LIVE primary would always
        # match (it is the value being edited), so the caller stashes the
        # as-built size under `_built` before any edits.
        built = int(self.cfg.get("_built", {}).get(
            "primary_team_count", self.cfg["league"]["primary_team_count"]))
        if self.n_teams != built:
            bad.append(f"tiers (cut for {built} teams)")
        return bad

    # -- draft bookkeeping -------------------------------------------------
    def picks_made(self, st: DraftState) -> int:
        return st.override if st.override is not None else len(st.order)

    def on_clock(self, st: DraftState) -> int:
        return self.picks_made(st) + 1

    def turns(self, st: DraftState) -> tuple[int | None, int | None]:
        """(this turn of yours, your next turn) as overall pick numbers."""
        clock = self.on_clock(st)
        later = [p for p in self.my_pick_nos if p >= clock]
        this = later[0] if later else None
        nxt = later[1] if len(later) > 1 else None
        if nxt is None and this is not None:
            nxt = this + self.n_teams      # past the end: assume one more turn
        return this, nxt

    def available(self, st: DraftState) -> np.ndarray:
        mask = np.ones(len(self.board), dtype=bool)
        if st.order:
            mask[list(st.drafted)] = False
        return mask

    # -- availability model ------------------------------------------------
    def _survival(self, rows: np.ndarray, next_turn: int) -> np.ndarray:
        """P(this player is still there at `next_turn`), modelling his realised
        draft slot as Normal(ADP, ADP_sd)."""
        if len(rows) == 0:
            return np.array([])
        s = 1.0 - norm.cdf(next_turn, loc=self.adp[rows], scale=self.sd[rows])
        return np.clip(s, 0.0, 1.0)

    def exp_best_next(self, st: DraftState, pos: str,
                      next_turn: int | None = None) -> float:
        """Expected VBD of the best player at `pos` still on the board at your
        next turn: sum_i vbd_i * P(i survives) * prod_{j better}(1 - P(j)).

        The product term is the chance every better player is already gone, so
        each player is weighted by the odds he is the best one left."""
        if next_turn is None:
            _, next_turn = self.turns(st)
        if next_turn is None:
            return 0.0
        rows = self.pos_queue(st, pos, DEPTH)
        if len(rows) == 0:
            return 0.0
        surv = self._survival(rows, next_turn)
        surv = np.where(self.vbd[rows] == 0, 0.0, surv)
        carry = np.concatenate(([1.0], np.cumprod(1.0 - surv)[:-1]))
        return float(np.sum(self.vbd[rows] * surv * carry))

    def pos_queue(self, st: DraftState, pos: str, n: int) -> np.ndarray:
        """Board rows for the best `n` available players at a position, in VBD
        order (the board is pre-sorted, so this is just a filtered head)."""
        rows = self._pos_rows[pos]
        avail = self.available(st)
        return rows[avail[rows]][:n]

    def vona(self, st: DraftState) -> dict[str, dict]:
        """Per-position live view: best available, his VBD, and VONA."""
        _, next_turn = self.turns(st)
        mine_pos = [self.pos[i] for i in st.mine]
        out = {}
        for pos in FANTASY_POS:
            q = self.pos_queue(st, pos, DEPTH)
            best = int(q[0]) if len(q) else None
            vbd = float(self.vbd[best]) if best is not None else 0.0
            exp = self.exp_best_next(st, pos, next_turn)
            held = mine_pos.count(pos)
            full = held >= self.useful[pos]
            out[pos] = {
                "row": best,
                "name": self.names[best] if best is not None else "-",
                "vbd": vbd,
                "exp_best_next": exp,
                "vona": vbd - exp,
                "yours": held,
                "full": full,
                "survival": float(self._survival(np.array([best]), next_turn)[0])
                if best is not None and next_turn else 0.0,
            }
        return out

    def recommend(self, st: DraftState) -> dict:
        """The single headline call. Positions you can no longer use are
        excluded outright rather than left to win on raw VONA - without that,
        QB VONA stays huge late and a 1QB roster collects quarterbacks."""
        v = self.vona(st)
        usable = {p: d for p, d in v.items()
                  if not d["full"] and d["row"] is not None}
        if not usable:
            return {"pos": None, "name": None, "vona": None,
                    "headline": "ROSTER FULL - take best available",
                    "positions": v}
        pos = max(usable, key=lambda p: usable[p]["vona"])
        d = usable[pos]
        return {"pos": pos, "row": d["row"], "name": d["name"],
                "vona": d["vona"],
                "headline": f"TAKE: {d['name']}  ({pos}, {d['vona']:+.1f} VONA)",
                "positions": v}

    # -- scarcity ----------------------------------------------------------
    def scarcity(self, st: DraftState) -> dict[str, dict]:
        avail = self.available(st)
        out = {}
        for pos in FANTASY_POS:
            rows = self._pos_rows[pos]
            rows = rows[avail[rows]]
            if len(rows) == 0:
                out[pos] = {"left": 0, "best_tier": None, "in_tier": 0,
                            "tier_size": 0, "drying": False}
                continue
            best_tier = int(self.tier[rows].min())
            in_tier = int((self.tier[rows] == best_tier).sum())
            # How big that tier was before the draft started.
            full_rows = self._pos_rows[pos]
            tier_size = int((self.tier[full_rows] == best_tier).sum())
            out[pos] = {
                "left": int(len(rows)),
                "best_tier": best_tier,
                "in_tier": in_tier,
                "tier_size": tier_size,
                # The workbook flagged any tier with <=2 left, which screams
                # from pick 1 whenever a top tier is naturally small (QB tier 1
                # has two players in it). A cliff only matters once it is
                # actually being eaten, so also require that some of this tier
                # has already gone.
                "drying": in_tier <= 2 and in_tier < tier_size,
            }
        return out

    def runs(self, st: DraftState, window: int | None = None) -> dict[str, int]:
        """How many of the last `window` picks went to each position. The
        spreadsheet could not do this - it only ever saw a set of marks, with
        no ordering - so a genuine run detector is new here."""
        w = window or self.run_window
        recent = st.order[-w:]
        counts = {p: 0 for p in FANTASY_POS}
        for pick in recent:
            p = self.pos[pick.idx]
            if p in counts:
                counts[p] += 1
        return counts

    # -- roster ------------------------------------------------------------
    def roster(self, st: DraftState) -> list[int]:
        return st.mine

    def roster_needs(self, st: DraftState) -> dict[str, int]:
        """Starter slots still unfilled, ignoring FLEX."""
        held = [self.pos[i] for i in st.mine]
        need = {}
        for p, want in self.cfg["league"]["roster"].items():
            if p == "FLEX":
                continue
            need[p] = max(0, int(want) - held.count(p))
        return need
