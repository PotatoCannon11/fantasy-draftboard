#!/usr/bin/env python
"""Keyboard-driven draft dashboard.

Everything here is reachable without the mouse - on the clock you should never
have to look for a pointer. The engine is `src/live.py`, which `test_live.py`
holds to parity with the workbook, so the xlsx remains a working backup.

    draftboard                       # your configured slot
    draftboard --slot 7 --teams 12
    draftboard --config              # open setup first
    draftboard --sleeper <draft_id>  # mirror a live Sleeper draft
    draftboard --reset               # discard a saved draft

Draft state autosaves to output/draft_state.json after every keystroke that
changes it, so closing the terminal mid-draft costs nothing.

Keys
  j/k ↑/↓ move      g/G top/bottom     PgUp/PgDn page
  d       someone else drafted him     m  I drafted him
  u       undo last pick               x  unmark the player under the cursor
  /       search (Esc clears)          a  all positions
  1-6     filter QB/RB/WR/TE/K/DEF
  Enter   jump to the recommended player
  c       add/remove from compare      C  compare view
  o       override the pick count      ,  setup / config
  R       reload board + news from disk
  F       fetch fresh data (news / market / everything)
  ?       help                         q  quit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    DataTable, Footer, Header, Input, Static,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    CONFIG, DATA_PROC, DATA_RAW, FANTASY_POS, OUTPUT, ensure_dirs,
    load_config,
)
from common import ROOT  # noqa: E402
from live import DraftState, LiveBoard, Pick  # noqa: E402
from notes import apply_news  # noqa: E402

STATE_PATH = OUTPUT / "draft_state.json"

POS_KEYS = {"1": "QB", "2": "RB", "3": "WR", "4": "TE", "5": "K", "6": "DEF"}
FLAG_STYLE = {"out": "bold red", "down": "dark_orange",
              "watch": "yellow", "up": "green"}


# Everything the board is derived from. Re-read on demand so a mid-draft news
# pass or a fresh pipeline run can be picked up without losing the draft.
SOURCES = [
    DATA_PROC / "draft_board.parquet",
    CONFIG / "news.yaml",
    CONFIG / "weights.yaml",
    DATA_RAW / "news_auto.parquet",
]


def source_stamp() -> tuple:
    """Mtimes of every input, so a change on disk can be noticed."""
    return tuple(p.stat().st_mtime if p.exists() else 0.0 for p in SOURCES)


def load_board(cfg: dict) -> pd.DataFrame:
    """The same universe the workbook prints, so the two agree."""
    b = pd.read_parquet(DATA_PROC / "draft_board.parquet")
    b = b.sort_values("vbd_score", ascending=False).reset_index(drop=True)
    b["overall_rank"] = range(1, len(b) + 1)
    o = cfg["output"]
    keep = b["n_sources"].fillna(0) >= 2
    keep |= b["overall_rank"] <= o.get("single_source_rank_grace", 60)
    if o.get("single_source_keep_if_adp", True):
        keep |= b[f"adp_{o['adp_teams']}"].notna()
    b = b[keep].head(o["board_depth"]).reset_index(drop=True)
    b["overall_rank"] = range(1, len(b) + 1)
    # `notes` / `news_flag` are NOT stored in the parquet - they are derived
    # from news.yaml + the auto-news cache + injury flags at render time. Skip
    # this and the entire news pass is invisible.
    return apply_news(b)


class Headline(Static):
    """The single call, big and unmissable."""


class Scarcity(Static):
    pass


class Detail(Static):
    pass


class HelpScreen(ModalScreen):
    BINDINGS = [Binding("escape,question_mark,q", "dismiss", "close")]

    def compose(self) -> ComposeResult:
        yield Static(__doc__[__doc__.index("Keys"):], id="helpbox")


class CompareScreen(ModalScreen):
    BINDINGS = [Binding("escape,C,q", "dismiss", "close")]

    def __init__(self, board, rows):
        super().__init__()
        self.board, self.rows = board, rows

    def compose(self) -> ComposeResult:
        if not self.rows:
            yield Static("Nothing to compare - press c on a player first.",
                         id="helpbox")
            return
        fields = [("Pos", "position"), ("Team", "team"), ("Tier", "tier"),
                  ("Proj", "final_projection"), ("VBD", "vbd_score"),
                  ("Floor", "projection_low"), ("Ceiling", "projection_high"),
                  ("ADP", "adp_10"), ("Val (rds)", "value_rounds"),
                  ("Bye", "bye_week"), ("SOS", "sos_z"), ("Age", "age"),
                  ("GmMiss3y", "games_missed_l3y")]
        t = DataTable(id="cmp")
        t.add_column("", width=12)
        picks = [self.board.iloc[i] for i in self.rows]
        for p in picks:
            t.add_column(str(p.player_name)[:18], width=19)
        # Highlight the best value per row so the eye lands on it immediately.
        higher_better = {"final_projection", "vbd_score", "projection_low",
                         "projection_high", "value_rounds"}
        lower_better = {"adp_10", "tier", "age", "games_missed_l3y"}
        for label, col in fields:
            vals = [p.get(col) for p in picks]
            nums = [v if isinstance(v, (int, float)) and pd.notna(v) else None
                    for v in vals]
            best = None
            real = [n for n in nums if n is not None]
            if real and col in higher_better:
                best = max(real)
            elif real and col in lower_better:
                best = min(real)
            cells = []
            for v, n in zip(vals, nums):
                s = "-" if v is None or (isinstance(v, float) and pd.isna(v)) \
                    else (f"{v:.1f}" if isinstance(v, float) else str(v))
                if best is not None and n == best:
                    s = f"[bold green]{s}[/]"
                cells.append(s)
            t.add_row(label, *cells)
        yield t


# Every knob the dashboard can change at the table, and whether the change is
# exact. LIVE fields recompute instantly off the loaded board. REBUILD fields
# feed numbers the pipeline baked in (replacement level -> VBD -> tiers), so
# changing them here shifts the roster maths but leaves VBD as-built; the screen
# says so rather than pretending otherwise.
CONFIG_FIELDS = [
    ("draft",  "Your draft slot",      ("simulation", "draft_slot"),      1, 1, 20, "live"),
    ("draft",  "Teams in the league",  ("league", "primary_team_count"),  1, 2, 20, "live"),
    ("draft",  "Rounds",               ("simulation", "rounds"),          1, 1, 30, "live"),
    ("roster", "QB starters",          ("league", "roster", "QB"),        1, 0, 4, "rebuild"),
    ("roster", "RB starters",          ("league", "roster", "RB"),        1, 0, 6, "rebuild"),
    ("roster", "WR starters",          ("league", "roster", "WR"),        1, 0, 6, "rebuild"),
    ("roster", "TE starters",          ("league", "roster", "TE"),        1, 0, 4, "rebuild"),
    ("roster", "FLEX",                 ("league", "roster", "FLEX"),      1, 0, 4, "rebuild"),
    ("roster", "K",                    ("league", "roster", "K"),         1, 0, 2, "rebuild"),
    ("roster", "DEF",                  ("league", "roster", "DEF"),       1, 0, 2, "rebuild"),
    ("roster", "Bench slots",          ("league", "bench_slots"),         1, 0, 15, "live"),
    ("assist", "Max useful QB",        ("output", "useful_max", "QB"),    1, 1, 6, "live"),
    ("assist", "Max useful RB",        ("output", "useful_max", "RB"),    1, 1, 10, "live"),
    ("assist", "Max useful WR",        ("output", "useful_max", "WR"),    1, 1, 10, "live"),
    ("assist", "Max useful TE",        ("output", "useful_max", "TE"),    1, 1, 6, "live"),
    ("assist", "Max useful K",         ("output", "useful_max", "K"),     1, 1, 3, "live"),
    ("assist", "Max useful DEF",       ("output", "useful_max", "DEF"),   1, 1, 3, "live"),
    ("assist", "Run-detector window",  ("output", "positional_run_window"), 1, 2, 30, "live"),
]
GROUP_NAMES = {"draft": "DRAFT", "roster": "ROSTER (needs pipeline re-run)",
               "assist": "LIVE ASSISTANT"}


def cfg_get(cfg: dict, path: tuple):
    d = cfg
    for k in path:
        d = d[k]
    return d


def cfg_set(cfg: dict, path: tuple, val) -> None:
    d = cfg
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = val


class ConfigScreen(ModalScreen):
    """Pre-draft setup. Changes apply the moment you make them - the whole
    reason for leaving the spreadsheet, where team size or slot meant re-running
    the pipeline and rebuilding the workbook."""

    BINDINGS = [
        Binding("escape,ctrl+c", "close", "back to board"),
        Binding("j,down", "cursor(1)", "down", show=False),
        Binding("k,up", "cursor(-1)", "up", show=False),
        Binding("l,right,plus,equals_sign", "bump(1)", "increase"),
        Binding("h,left,minus", "bump(-1)", "decrease"),
        Binding("s", "save", "save to weights.yaml"),
        Binding("r", "reload", "revert to file"),
    ]

    def __init__(self, app_ref):
        super().__init__()
        self.app_ref = app_ref
        self.row = 0
        self.saved = ""

    def compose(self) -> ComposeResult:
        with Vertical(id="cfgbox"):
            yield Static("", id="cfghead")
            yield DataTable(id="cfg", cursor_type="row", zebra_stripes=True)
            yield Static("", id="cfgfoot")

    def on_mount(self) -> None:
        t = self.query_one("#cfg", DataTable)
        t.add_column("Setting", width=34)
        t.add_column("Value", width=9)
        t.add_column("Applies", width=9)
        self.redraw()
        t.focus()

    def redraw(self) -> None:
        t = self.query_one("#cfg", DataTable)
        keep = t.cursor_row or 0
        t.clear()
        cfg = self.app_ref.cfg
        last = None
        self.index: list[int] = []
        for i, (grp, label, path, *_rest) in enumerate(CONFIG_FIELDS):
            kind = _rest[-1]
            if grp != last:
                t.add_row(f"[bold]── {GROUP_NAMES[grp]} ──[/]", "", "")
                self.index.append(-1)
                last = grp
            tag = "[green]live[/]" if kind == "live" else "[yellow]rebuild[/]"
            t.add_row(label, f"[bold]{cfg_get(cfg, path)}[/]", tag)
            self.index.append(i)
        t.move_cursor(row=min(keep, len(self.index) - 1))
        if self.index and self.index[t.cursor_row or 0] < 0:
            self._snap(1)
        self.head()

    def head(self) -> None:
        lb = self.app_ref.lb
        stale = lb.stale_for()
        msg = ("[bold]Setup[/]   h/l or -/+ change · j/k move · "
               "s save to weights.yaml · r revert · Esc back\n"
               f"picks from slot {lb.slot} of {lb.n_teams}: "
               f"{', '.join(str(p) for p in lb.my_pick_nos[:6])}…")
        if stale:
            msg += ("\n[yellow]not exact at this team size: " + "; ".join(stale)
                    + " — run_pipeline.py --stage board to make it exact[/]")
        self.query_one("#cfghead", Static).update(msg)
        self.query_one("#cfgfoot", Static).update(
            self.saved or "[dim]changes apply immediately; "
            "'s' also writes them to config/weights.yaml[/]")

    def _field(self) -> int | None:
        t = self.query_one("#cfg", DataTable)
        r = t.cursor_row
        if r is None or r >= len(self.index) or self.index[r] < 0:
            return None
        return self.index[r]

    def _snap(self, d: int) -> None:
        """Move onto the nearest real field; group headers are not stops."""
        t = self.query_one("#cfg", DataTable)
        r = t.cursor_row or 0
        while 0 <= r < len(self.index) and self.index[r] < 0:
            r += d
        t.move_cursor(row=max(0, min(len(self.index) - 1, r)))

    def action_cursor(self, d: int) -> None:
        t = self.query_one("#cfg", DataTable)
        r = (t.cursor_row or 0) + d
        while 0 <= r < len(self.index) and self.index[r] < 0:
            r += d                       # skip straight over the group header
        if 0 <= r < len(self.index):
            t.move_cursor(row=r)

    def current_label(self) -> str | None:
        f = self._field()
        return None if f is None else CONFIG_FIELDS[f][1]

    def action_bump(self, d: int) -> None:
        f = self._field()
        if f is None:
            return
        _grp, _label, path, step, lo, hi, _kind = CONFIG_FIELDS[f]
        cur = int(cfg_get(self.app_ref.cfg, path))
        cfg_set(self.app_ref.cfg, path, max(lo, min(hi, cur + d * step)))
        self.saved = ""
        self.app_ref.reconfigure()
        self.redraw()

    def action_save(self) -> None:
        ok, msg = self.app_ref.save_config()
        self.saved = (f"[green]{msg}[/]" if ok else f"[bold red]{msg}[/]")
        self.head()

    def action_reload(self) -> None:
        self.app_ref.reload_config()
        self.saved = "[dim]reverted to config/weights.yaml[/]"
        self.redraw()

    def action_close(self) -> None:
        self.dismiss(None)


# What `F` can pull, cheapest first. Each entry is (label, description, argv).
# Anything that re-runs the model moves VBD and tiers underneath you mid-draft,
# so the cost and the consequence are both spelled out in the menu.
# NOTE: `--stage features` SKIPS the ingest (run_pipeline drops stages before
# the named one, and ingest is first), so a re-pull must start at the default
# stage and be narrowed with --refresh instead. Getting this wrong silently
# recomputes from cache and pulls nothing.
FETCH_MODES = [
    ("n", "News",
     "Sleeper trending + ESPN headlines → Notes. ~5s, model untouched.",
     ["refresh_news.py"]),
    ("m", "Market — ADP + projections",
     "FantasyFootballCalculator ADP, plus Sleeper / ESPN / FantasySharks "
     "projections, then re-runs the model. ~60s — VBD, tiers and ADP move.",
     ["run_pipeline.py", "--refresh", "market"]),
    ("s", "Statistics — nflverse",
     "Rosters, depth charts, snap counts, NGS, play-by-play, weekly and "
     "seasonal stats, ffopportunity expected points, DynastyProcess ids. "
     "Then re-runs the model. ~3min.",
     ["run_pipeline.py", "--refresh", "nflverse"]),
    ("a", "Everything",
     "Both of the above — every dataset, then the whole pipeline. "
     "~4min — do NOT start this on the clock.",
     ["run_pipeline.py"]),
]


class FetchScreen(ModalScreen):
    """Pick what to re-fetch from the upstream datasets."""

    BINDINGS = [Binding("escape,q", "dismiss", "cancel")]

    def compose(self) -> ComposeResult:
        lines = ["[bold]Fetch fresh data[/]", ""]
        for key, label, desc, _argv in FETCH_MODES:
            lines += [f"  [bold yellow]{key}[/]  {label}", f"       [dim]{desc}[/]"]
        lines += ["", "[dim]Runs in the background — the board stays usable, "
                  "and your picks are kept. Esc to cancel.[/]"]
        yield Static("\n".join(lines), id="helpbox")

    def on_key(self, event) -> None:
        for key, _label, _desc, argv in FETCH_MODES:
            if event.key == key:
                self.dismiss(argv)
                event.stop()
                return


class OverrideScreen(ModalScreen):
    BINDINGS = [Binding("escape", "dismiss", "cancel")]

    def compose(self) -> ComposeResult:
        yield Input(placeholder="true pick count (blank = use marks)",
                    id="ovr")

    @on(Input.Submitted)
    def submit(self, ev: Input.Submitted) -> None:
        self.dismiss(ev.value.strip())


class DraftApp(App):
    CSS = """
    Screen { layout: vertical; }
    #headline { height: 3; content-align: center middle; text-style: bold;
                background: $primary; color: $text; }
    #scarcity { height: 5; border: round $accent; padding: 0 1; }
    #body { height: 1fr; }
    #board { width: 2fr; }
    #detail { width: 1fr; border: round $accent; padding: 0 1; overflow-y: auto; }
    #search { height: 3; display: none; }
    #search.on { display: block; }
    #helpbox { padding: 1 2; background: $surface; border: round $accent; }
    #cmp { padding: 1; background: $surface; }
    #cfgbox { width: 62; height: auto; max-height: 90%; padding: 1 2;
              background: $surface; border: round $accent; }
    #cfghead { height: auto; padding-bottom: 1; }
    #cfgfoot { height: auto; padding-top: 1; }
    #cfg { height: auto; max-height: 26; }
    """

    BINDINGS = [
        Binding("j,down", "move(1)", "down", show=False),
        Binding("k,up", "move(-1)", "up", show=False),
        Binding("g", "top", "top", show=False),
        Binding("G", "bottom", "bottom", show=False),
        Binding("d", "draft(False)", "drafted"),
        Binding("m", "draft(True)", "MINE"),
        Binding("u", "undo", "undo"),
        Binding("x", "unmark", "unmark"),
        Binding("slash", "search", "search"),
        Binding("a", "filter_all", "all"),
        # DataTable swallows plain enter for row selection, so this binding has
        # to outrank the focused widget or the jump silently does nothing.
        Binding("enter", "goto_rec", "go to pick", priority=True),
        Binding("asterisk", "goto_rec", "go to pick", show=False),
        Binding("c", "compare_toggle", "compare±"),
        Binding("C", "compare_show", "compare"),
        Binding("o", "override", "override"),
        Binding("comma", "config", "setup"),
        Binding("R", "reload_data", "reload"),
        Binding("F", "fetch", "fetch"),
        Binding("question_mark", "help", "help"),
        Binding("q", "quit", "quit"),
        *[Binding(k, f"filter('{p}')", p, show=False)
          for k, p in POS_KEYS.items()],
    ]

    filter_pos: reactive[str | None] = reactive(None)
    query: reactive[str] = reactive("")

    def __init__(self, cfg, slot=None, reset=False, sleeper=None,
                 interval=5.0, open_config=False):
        super().__init__()
        if slot:
            cfg["simulation"]["draft_slot"] = slot
        # Snapshot the size the pipeline actually built, before the setup
        # screen can edit it - stale_for() needs the as-built value to compare
        # against, not the value being edited.
        cfg.setdefault("_built", {"primary_team_count":
                                  int(cfg["league"]["primary_team_count"])})
        self.cfg = cfg
        self.board = load_board(cfg)
        self.lb = LiveBoard(self.board, cfg)
        self.board = self.lb.board          # engine sorts; keep one frame
        self.name_to_idx = {str(n): i for i, n in
                            enumerate(self.board.player_name)}
        self.st = DraftState()
        if not reset:
            self._load_state()
        self.compare: list[int] = []
        self.shown: list[int] = []
        self.open_config = open_config
        # -- sleeper --
        self.sleeper_id = sleeper
        self.sleeper_interval = interval
        self.sleeper_status = "" if not sleeper else "connecting…"
        self.sleeper_unmatched: list[str] = []
        self._bmap = None
        # -- data freshness --
        self.stamp = source_stamp()
        self.stale = False
        self.fetching = ""
        self.fetch_msg = ""

    # -- reloading from the datasets ---------------------------------------
    def reload_data(self) -> str:
        """Re-read the board and news from disk and rebuild the engine.

        Draft state is stored as row indices, and a reload can reorder or
        resize the board, so the picks are remapped through player NAMES.
        Skipping that silently reassigns every pick to a different player."""
        keep = [(str(self.board.player_name[p.idx]), p.mine)
                for p in self.st.order]
        keep_cmp = [str(self.board.player_name[i]) for i in self.compare]
        override = self.st.override

        self.board = load_board(self.cfg)
        self.lb = LiveBoard(self.board, self.cfg)
        self.board = self.lb.board
        self.name_to_idx = {str(n): i for i, n in
                            enumerate(self.board.player_name)}
        self._bmap = None                 # rebuilt against the new board

        self.st = DraftState(override=override)
        lost = []
        for nm, mine in keep:
            i = self.name_to_idx.get(nm)
            if i is None:
                lost.append(nm)
            else:
                self.st.order.append(Pick(i, mine))
        self.compare = [self.name_to_idx[n] for n in keep_cmp
                        if n in self.name_to_idx]
        self.stamp = source_stamp()
        self.stale = False
        self.refresh_all()
        n_notes = int((self.board["notes"].astype(str) != "").sum())
        msg = f"reloaded — {len(self.board)} players, {n_notes} notes"
        if lost:
            msg += f"; {len(lost)} pick(s) no longer on the board: " \
                   + ", ".join(lost[:3])
        return msg

    def check_sources(self) -> None:
        """Notice when the data changes underneath us, so the dashboard can
        never be quietly running on a board that has since been rebuilt."""
        now = source_stamp()
        if now != self.stamp and not self.stale:
            self.stale = True
            self.refresh_panels()

    async def run_fetch(self, argv: list[str]) -> None:
        """Re-fetch upstream datasets in a subprocess, then reload.

        Deliberately out-of-process and non-blocking: a pull can take minutes
        and the board has to stay usable while it runs."""
        import asyncio
        if self.fetching:
            return
        self.fetching = " ".join(argv[:1])
        self.refresh_panels()
        try:
            here = Path(__file__).resolve().parent
            cmd = [str(here / a) if a.endswith(".py") else a for a in argv]
            proc = await asyncio.create_subprocess_exec(
                sys.executable, *cmd, cwd=str(ROOT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT)
            out, _ = await proc.communicate()
            if proc.returncode != 0:
                tail = (out or b"").decode(errors="replace").strip()\
                    .splitlines()[-1:]
                self.fetch_msg = f"[red]fetch failed: {' '.join(tail)}[/]"
            else:
                self.fetch_msg = f"[green]{self.reload_data()}[/]"
        except Exception as exc:  # noqa: BLE001
            self.fetch_msg = f"[red]fetch failed: {exc}[/]"
        finally:
            self.fetching = ""
            self.refresh_all()

    # -- live reconfiguration ---------------------------------------------
    def reconfigure(self) -> None:
        """Rebuild the engine from the current config. Cheap - it is numpy
        views over an already-loaded frame - which is what lets the setup
        screen apply changes instantly instead of needing a pipeline run."""
        self.lb = LiveBoard(self.board, self.cfg)
        self.board = self.lb.board
        self.refresh_all()

    def save_config(self) -> tuple[bool, str]:
        """Persist the live config back into config/weights.yaml, preserving
        everything the dashboard does not manage."""
        import yaml
        path = CONFIG / "weights.yaml"
        try:
            doc = yaml.safe_load(path.read_text()) or {}
            for _grp, _label, fpath, *_r in CONFIG_FIELDS:
                cfg_set(doc, fpath, cfg_get(self.cfg, fpath))
            path.write_text(yaml.safe_dump(doc, sort_keys=False,
                                           default_flow_style=False))
        except (OSError, ValueError, KeyError) as exc:
            return False, f"could not save: {exc}"
        return True, (f"saved to {path.name} — note VBD/tiers still come from "
                      "the last pipeline run")

    def reload_config(self) -> None:
        built = self.cfg.get("_built")
        self.cfg = load_config()
        if built:
            self.cfg["_built"] = built
        self.reconfigure()

    # -- persistence -------------------------------------------------------
    def _save_state(self) -> None:
        STATE_PATH.write_text(json.dumps({
            "order": [{"name": str(self.board.player_name[p.idx]),
                       "mine": p.mine} for p in self.st.order],
            "override": self.st.override,
        }, indent=1))

    def _load_state(self) -> None:
        if not STATE_PATH.exists():
            return
        try:
            doc = json.loads(STATE_PATH.read_text())
        except (OSError, ValueError):
            return
        idx = {str(n): i for i, n in enumerate(self.board.player_name)}
        for p in doc.get("order", []):
            i = idx.get(p.get("name"))
            if i is not None:
                self.st.order.append(Pick(i, bool(p.get("mine"))))
        self.st.override = doc.get("override")

    # -- layout ------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header()
        yield Headline(id="headline")
        yield Scarcity(id="scarcity")
        with Horizontal(id="body"):
            yield DataTable(id="board", cursor_type="row", zebra_stripes=True)
            yield Detail(id="detail")
        yield Input(placeholder="search player…", id="search")
        yield Footer()

    def on_mount(self) -> None:
        t = self.query_one("#board", DataTable)
        for label, w in (("Rk", 4), ("Player", 22), ("P", 4), ("Tm", 4),
                         ("Tier", 5), ("VBD", 7), ("ADP", 6), ("Val", 6),
                         ("Bye", 4), ("Notes", 40)):
            t.add_column(label, width=w)
        self.refresh_all()
        t.focus()
        self.set_interval(3.0, self.check_sources)
        if self.sleeper_id:
            self.set_interval(self.sleeper_interval, self.poll_sleeper)
            self.call_after_refresh(self.poll_sleeper)
        if self.open_config:
            self.call_after_refresh(self.action_config)

    # -- sleeper ------------------------------------------------------------
    async def poll_sleeper(self) -> None:
        """Pull the draft and mirror it into local state.

        The feed is authoritative while connected: it is an ordered pick list,
        so the local order is replaced wholesale rather than merged. That keeps
        undo-on-Sleeper's-side correct, at the cost of discarding manual marks -
        which is right, because the Sleeper draft IS the draft."""
        import asyncio

        from sleeper_sync import BoardMap, fetch_picks
        if self._bmap is None:
            self._bmap = BoardMap(board=self.board)
        try:
            picks = await asyncio.to_thread(fetch_picks, self.sleeper_id)
        except Exception as exc:  # noqa: BLE001
            self.sleeper_status = f"[red]offline — retrying ({exc.__class__.__name__})[/]"
            self.refresh_panels()
            return

        order, unmatched = [], []
        for p in picks:
            nm = self._bmap.resolve_name(p)
            i = self.name_to_idx.get(nm) if nm else None
            if i is None:
                unmatched.append(f"#{p.get('pick_no')} {self._bmap.describe(p)}")
                continue
            order.append(Pick(i, mine=int(p.get("draft_slot") or -1)
                              == self.lb.slot))
        changed = [p.idx for p in order] != [p.idx for p in self.st.order]
        self.st.order = order
        self.sleeper_unmatched = unmatched
        miss = f"  [yellow]{len(unmatched)} unmatched[/]" if unmatched else ""
        self.sleeper_status = f"[green]sleeper ●[/] {len(order)} picks{miss}"
        if changed:
            self._save_state()
        self.refresh_all()

    # -- rendering ---------------------------------------------------------
    def _rows(self) -> list[int]:
        b = self.board
        mask = pd.Series(True, index=b.index)
        if self.filter_pos:
            mask &= b.position == self.filter_pos
        if self.query:
            mask &= b.player_name.str.contains(self.query, case=False,
                                               na=False, regex=False)
        return list(b.index[mask])

    def refresh_all(self) -> None:
        self.refresh_table()
        self.refresh_panels()

    def refresh_table(self) -> None:
        t = self.query_one("#board", DataTable)
        keep = t.cursor_row
        t.clear()
        drafted, mine = self.st.drafted, set(self.st.mine)
        rec = self.lb.recommend(self.st)
        rec_row = rec.get("row")
        self.shown = self._rows()
        for i in self.shown:
            r = self.board.iloc[i]
            note = str(r.get("notes") or "")[:38]
            flag = str(r.get("news_flag") or "")
            if flag in FLAG_STYLE and note:
                note = f"[{FLAG_STYLE[flag]}]{note}[/]"
            name = str(r.player_name)
            if i in mine:
                name = f"[bold green]★ {name}[/]"
            elif i in drafted:
                name = f"[strike dim]{name}[/]"
            elif i == rec_row:
                name = f"[bold yellow]▶ {name}[/]"
            if i in self.compare:
                name = f"[on #333333]{name}[/]"
            val = r.get("value_rounds")
            t.add_row(
                str(int(r.overall_rank)), name, str(r.position), str(r.team),
                str(r.get("tier_label") or ""),
                f"{r.vbd_score:.1f}" if pd.notna(r.vbd_score) else "-",
                f"{r.adp_10:.0f}" if pd.notna(r.get("adp_10")) else "-",
                f"{val:+.2f}" if pd.notna(val) else "-",
                str(int(r.bye_week)) if pd.notna(r.get("bye_week")) else "-",
                note or "",
            )
        if keep is not None and self.shown:
            t.move_cursor(row=min(keep, len(self.shown) - 1))

    # Panel text is built by pure functions so it can be asserted on directly
    # (Textual's Static does not expose its rendered content for inspection).
    def headline_text(self) -> str:
        rec = self.lb.recommend(self.st)
        this, nxt = self.lb.turns(self.st)
        rnd = self.lb.picks_made(self.st) // self.lb.n_teams + 1
        ovr = "  [OVERRIDE]" if self.st.override is not None else ""
        sl = f"  ·  {self.sleeper_status}" if self.sleeper_status else ""
        return (f"{rec['headline']}     ·  pick #{self.lb.on_clock(self.st)} "
                f"(R{rnd})  ·  your turn {this}  ·  next {nxt}{ovr}{sl}")

    def scarcity_text(self) -> str:
        rec = self.lb.recommend(self.st)
        sc = self.lb.scarcity(self.st)
        runs = self.lb.runs(self.st)
        v = rec["positions"]
        line1, line2 = [], []
        for p in FANTASY_POS:
            s, d = sc[p], v[p]
            tag = f"{p} {s['left']:>3} T{s['best_tier']}({s['in_tier']})"
            line1.append(f"[bold red]{tag}![/]"
                         if s["drying"] and s["left"] else tag)
            vona = f"{d['vona']:+.0f}"
            if d["full"]:
                lab = f"[dim]{p} FULL[/]"
            elif p == rec["pos"]:
                lab = f"[bold yellow]{p} {vona}[/]"
            else:
                lab = f"{p} {vona}"
            line2.append(f"{lab}·{runs[p]}")
        out = ("left/tier   " + "  ".join(line1) + "\n"
               + "VONA·run    " + "  ".join(line2))
        if self.fetching:
            out += f"\n[yellow]fetching {self.fetching}… board still usable[/]"
        elif self.stale:
            out += ("\n[bold yellow]data on disk changed — press R to "
                    "reload[/]")
        elif self.fetch_msg:
            out += f"\n{self.fetch_msg}"
        return out

    def refresh_panels(self) -> None:
        self.query_one("#headline", Headline).update(self.headline_text())
        self.query_one("#scarcity", Scarcity).update(self.scarcity_text())
        self.refresh_detail()

    def refresh_detail(self) -> None:
        t = self.query_one("#board", DataTable)
        d = self.query_one("#detail", Detail)
        if t.cursor_row is None or t.cursor_row >= len(self.shown):
            d.update("")
            return
        r = self.board.iloc[self.shown[t.cursor_row]]
        needs = self.lb.roster_needs(self.st)
        roster = [f"{self.board.position[i]} {self.board.player_name[i]}"
                  for i in self.st.mine]

        def g(c, fmt="{:.1f}"):
            v = r.get(c)
            return fmt.format(v) if isinstance(v, (int, float)) and pd.notna(v) \
                else "-"
        lines = [
            f"[bold]{r.player_name}[/]  {r.position} · {r.team}",
            f"tier {r.get('tier_label')}  ({r.get('sub_tier_label')})",
            "",
            f"proj    {g('final_projection')}   "
            f"floor {g('projection_low')}  ceil {g('projection_high')}",
            f"VBD     {g('vbd_score')}   (8tm {g('vbd_8')} / 10tm {g('vbd_10')})",
            f"ADP     {g('adp_10','{:.0f}')}   value {g('value_rounds','{:+.2f}')} rds",
            f"bye     {g('bye_week','{:.0f}')}    SOS {g('sos_z','{:+.2f}')}",
            f"age     {g('age')}    missed {g('games_missed_l3y','{:.0f}')} gm/3y",
            f"sources {g('n_sources','{:.0f}')}    ctx {g('context_multiplier','{:.2f}')}",
        ]
        note = str(r.get("notes") or "")
        if note:
            flag = str(r.get("news_flag") or "")
            style = FLAG_STYLE.get(flag, "")
            lines += ["", f"[{style}]{note}[/]" if style else note]
        lines += ["", f"[bold]your roster[/] ({len(roster)})"]
        lines += [f"  {x}" for x in roster] or ["  (empty)"]
        short = [f"{p}×{n}" for p, n in needs.items() if n]
        lines += ["", "still need: " + (", ".join(short) if short
                                        else "starters filled")]
        if self.compare:
            lines += ["", "compare: " + ", ".join(
                str(self.board.player_name[i]) for i in self.compare)]
        d.update("\n".join(lines))

    # -- actions -----------------------------------------------------------
    def _cursor_idx(self) -> int | None:
        t = self.query_one("#board", DataTable)
        if t.cursor_row is None or t.cursor_row >= len(self.shown):
            return None
        return self.shown[t.cursor_row]

    def action_move(self, delta: int) -> None:
        t = self.query_one("#board", DataTable)
        t.move_cursor(row=max(0, min(len(self.shown) - 1,
                                     (t.cursor_row or 0) + delta)))
        self.refresh_detail()

    def action_top(self) -> None:
        self.query_one("#board", DataTable).move_cursor(row=0)
        self.refresh_detail()

    def action_bottom(self) -> None:
        self.query_one("#board", DataTable).move_cursor(
            row=max(0, len(self.shown) - 1))
        self.refresh_detail()

    def action_draft(self, mine: bool) -> None:
        i = self._cursor_idx()
        if i is None or i in self.st.drafted:
            return
        self.st.take(i, mine=mine)
        self._save_state()
        self.refresh_all()

    def action_undo(self) -> None:
        if self.st.undo():
            self._save_state()
            self.refresh_all()

    def action_unmark(self) -> None:
        i = self._cursor_idx()
        if i is not None and i in self.st.drafted:
            self.st.unmark(i)
            self._save_state()
            self.refresh_all()

    def action_filter(self, pos: str) -> None:
        self.filter_pos = None if self.filter_pos == pos else pos
        self.refresh_table()
        self.refresh_detail()

    def action_filter_all(self) -> None:
        self.filter_pos = None
        self.query = ""
        self.query_one("#search", Input).value = ""
        self.query_one("#search").remove_class("on")
        self.refresh_table()
        self.refresh_detail()

    def action_search(self) -> None:
        s = self.query_one("#search", Input)
        s.add_class("on")
        s.focus()

    def action_goto_rec(self) -> None:
        # This binding is priority, so it also fires while the search box has
        # focus. There, enter means "done typing" - without this the Input keeps
        # focus and every later keystroke is swallowed as search text.
        if isinstance(self.focused, Input):
            self.query_one("#board", DataTable).focus()
            return
        rec = self.lb.recommend(self.st)
        row = rec.get("row")
        if row is None:
            return
        if row not in self.shown:
            self.filter_pos = None
            self.query = ""
            self.refresh_table()
        if row in self.shown:
            self.query_one("#board", DataTable).move_cursor(
                row=self.shown.index(row))
            self.refresh_detail()

    def action_compare_toggle(self) -> None:
        i = self._cursor_idx()
        if i is None:
            return
        if i in self.compare:
            self.compare.remove(i)
        elif len(self.compare) < 4:
            self.compare.append(i)
        self.refresh_table()
        self.refresh_detail()

    def action_compare_show(self) -> None:
        self.push_screen(CompareScreen(self.board, self.compare))

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_config(self) -> None:
        self.push_screen(ConfigScreen(self))

    def action_reload_data(self) -> None:
        self.fetch_msg = f"[green]{self.reload_data()}[/]"
        self.refresh_panels()

    def action_fetch(self) -> None:
        def go(argv: list[str] | None) -> None:
            if argv:
                self.run_worker(self.run_fetch(argv), exclusive=False)
        self.push_screen(FetchScreen(), go)

    def action_override(self) -> None:
        def done(val: str | None) -> None:
            if val is None:
                return
            self.st.override = int(val) if val.isdigit() else None
            self._save_state()
            self.refresh_all()
        self.push_screen(OverrideScreen(), done)

    @on(Input.Changed, "#search")
    def on_search(self, ev: Input.Changed) -> None:
        self.query = ev.value
        self.refresh_table()

    @on(Input.Submitted, "#search")
    def on_search_done(self, _ev) -> None:
        self.query_one("#board", DataTable).focus()

    @on(DataTable.RowHighlighted)
    def on_row(self, _ev) -> None:
        self.refresh_detail()


FIRST_RUN = """\
No draft board yet — nothing has been pulled for this data directory.

  data home : {home}
  build it  : draftboard-pipeline

That downloads the datasets (nflverse stats, ADP, projections) and builds the
board. It takes ~4 minutes and needs a network connection; afterwards
`draftboard` starts instantly and works offline.

Point FANTASY_HOME somewhere else to keep separate leagues apart.
"""


def main(argv=None) -> int:
    ensure_dirs()   # create the data home and seed config on a fresh install
    board_path = DATA_PROC / "draft_board.parquet"
    if not board_path.exists():
        # A brand-new install has config but no data. Say what to run rather
        # than dying in pandas with a bare FileNotFoundError.
        print(FIRST_RUN.format(home=ROOT))
        return 1
    cfg = load_config()
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slot", type=int, help="your seat (1 = first overall)")
    ap.add_argument("--teams", type=int, help="teams in the league")
    ap.add_argument("--reset", action="store_true",
                    help="discard any saved draft state")
    ap.add_argument("--config", action="store_true",
                    help="open the setup screen on launch")
    ap.add_argument("--sleeper", metavar="DRAFT_ID",
                    help="mirror a live Sleeper draft (id or URL)")
    ap.add_argument("--interval", type=float, default=5.0,
                    help="seconds between Sleeper polls (default 5)")
    args = ap.parse_args(argv)

    if args.reset and STATE_PATH.exists():
        STATE_PATH.unlink()
    if args.teams:
        cfg["league"]["primary_team_count"] = args.teams

    sleeper_id = None
    if args.sleeper:
        from sleeper_sync import fetch_draft, parse_draft_id
        sleeper_id = parse_draft_id(args.sleeper)
        # Take the room's real shape from the draft itself - guessing it wrong
        # silently corrupts every turn number and therefore every VONA.
        try:
            info = fetch_draft(sleeper_id)
            st = info.get("settings") or {}
            if st.get("teams"):
                cfg["league"]["primary_team_count"] = int(st["teams"])
            if st.get("rounds"):
                cfg["simulation"]["rounds"] = int(st["rounds"])
            print(f"sleeper draft {sleeper_id}: {info.get('status')}, "
                  f"{st.get('teams')} teams, {st.get('rounds')} rounds")
        except Exception as exc:  # noqa: BLE001
            print(f"could not read the Sleeper draft ({exc}). "
                  "Starting anyway; it will keep retrying.")

    DraftApp(cfg, slot=args.slot, reset=args.reset, sleeper=sleeper_id,
             interval=args.interval, open_config=args.config).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
