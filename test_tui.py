#!/usr/bin/env python
"""Drive the dashboard headlessly and assert every keyboard path works.

Textual's pilot presses real keys against a real app, so this covers the
bindings, the draft-state mutations behind them, and the panel text - the
things that would otherwise only be found on the clock.

    .venv/bin/python test_tui.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# src must win over the root shims, which only re-export these names
sys.path.insert(0, str(Path(__file__).resolve().parent / "fantasydraft"))

from common import load_config  # noqa: E402

import draft_tui as T  # noqa: E402

FAILS: list[str] = []
PASSES = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSES
    if cond:
        PASSES += 1
        print(f"  [pass] {name}")
    else:
        FAILS.append(f"{name}: {detail}")
        print(f"  [FAIL] {name}  {detail}")


async def run() -> None:
    cfg = load_config()
    if T.STATE_PATH.exists():
        T.STATE_PATH.unlink()
    app = T.DraftApp(cfg, reset=True)
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        board = app.board

        lb_teams_before = app.lb.n_teams

        print("=== startup ===")
        check("board loads the printed universe", len(app.shown) > 400,
              f"{len(app.shown)} rows")
        check("headline names a player on an empty board",
              "TAKE:" in app.headline_text(), app.headline_text()[:60])
        check("no tier is 'drying up' before a single pick",
              not any(d["drying"] for d in app.lb.scarcity(app.st).values()),
              str({p: d for p, d in app.lb.scarcity(app.st).items()
                   if d["drying"]}))

        print("\n=== marking ===")
        first = app.shown[0]
        await pilot.press("d")
        await pilot.pause()
        check("d marks the room's pick", first in app.st.drafted)
        check("d does not claim it as yours", first not in app.st.mine)
        await pilot.press("j", "m")
        await pilot.pause()
        second = app.shown[1]
        check("m claims a player as yours", second in app.st.mine)
        check("picks made counts both", app.lb.picks_made(app.st) == 2)
        check("headline moved off the drafted player",
              str(board.player_name[first]) not in app.headline_text(),
              app.headline_text()[:60])

        print("\n=== undo / unmark ===")
        await pilot.press("u")
        await pilot.pause()
        check("u undoes the last pick", second not in app.st.drafted)
        check("u leaves earlier picks alone", first in app.st.drafted)
        await pilot.press("g", "x")
        await pilot.pause()
        check("x unmarks the player under the cursor",
              first not in app.st.drafted)

        print("\n=== filters and search ===")
        await pilot.press("2")
        await pilot.pause()
        check("digit filters to one position",
              {board.position[i] for i in app.shown} == {"RB"},
              str({board.position[i] for i in app.shown}))
        await pilot.press("2")
        await pilot.pause()
        check("same digit toggles the filter off", len(app.shown) > 400)
        await pilot.press("4")
        await pilot.pause()
        check("a different digit filters to TE",
              {board.position[i] for i in app.shown} == {"TE"})
        await pilot.press("a")
        await pilot.pause()
        check("a clears back to everything", len(app.shown) > 400)

        target = str(board.player_name[app.shown[3]])
        await pilot.press("slash")
        for ch in target[:6]:
            await pilot.press(ch if ch != " " else "space")
        await pilot.pause()
        check("search narrows the board",
              0 < len(app.shown) < 400 and
              any(str(board.player_name[i]).startswith(target[:6])
                  for i in app.shown),
              f"{len(app.shown)} rows for {target[:6]!r}")
        await pilot.press("enter")
        await pilot.press("a")
        await pilot.pause()
        check("a resets the search too", len(app.shown) > 400)

        print("\n=== recommendation jump ===")
        await pilot.press("4")          # filter away from the recommendation
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        rec = app.lb.recommend(app.st)
        cur = app.shown[app.query_one("#board").cursor_row]
        check("enter jumps to the recommended player, clearing the filter",
              cur == rec["row"],
              f"landed on {board.player_name[cur]}, want {rec['name']}")

        print("\n=== compare ===")
        await pilot.press("c", "j", "c", "j", "c")
        await pilot.pause()
        check("c collects players to compare", len(app.compare) == 3,
              str(app.compare))
        await pilot.press("c")
        await pilot.pause()
        check("c on a selected player removes him", len(app.compare) == 2)
        await pilot.press("C")
        await pilot.pause()
        check("C opens the compare screen",
              isinstance(app.screen, T.CompareScreen), type(app.screen).__name__)
        await pilot.press("escape")
        await pilot.pause()
        check("escape closes it", not isinstance(app.screen, T.CompareScreen))

        print("\n=== help ===")
        await pilot.press("question_mark")
        await pilot.pause()
        check("? opens help", isinstance(app.screen, T.HelpScreen))
        await pilot.press("escape")
        await pilot.pause()
        check("escape closes help", not isinstance(app.screen, T.HelpScreen))

        print("\n=== roster awareness in the live panel ===")
        cap = int(cfg["output"]["useful_max"]["QB"])
        for i in [int(x) for x in board.index[board.position == "QB"][:cap]]:
            app.st.take(i, mine=True)
        app.refresh_all()
        await pilot.pause()
        check("a filled position reports FULL in the strip",
              "QB FULL" in app.scarcity_text(), app.scarcity_text())
        check("a filled position is not the headline call",
              app.lb.recommend(app.st)["pos"] != "QB")

        print("\n=== setup screen (live reconfiguration) ===")
        await pilot.press("comma")
        await pilot.pause()
        check(", opens setup", isinstance(app.screen, T.ConfigScreen),
              type(app.screen).__name__)
        slot_before = app.lb.slot
        picks_before = list(app.lb.my_pick_nos)
        await pilot.press("l")            # slot +1, first field
        await pilot.pause()
        check("changing a setting takes effect immediately",
              app.lb.slot == slot_before + 1,
              f"{app.lb.slot} vs {slot_before}")
        check("your pick numbers recompute on the spot",
              app.lb.my_pick_nos != picks_before,
              "snake picks did not move")
        await pilot.press("h")
        await pilot.pause()
        check("and back down again", app.lb.slot == slot_before)

        await pilot.press("j", "l")       # teams +1
        await pilot.pause()
        check("team count is live too",
              app.lb.n_teams == cfg["league"]["primary_team_count"])
        check("a team size the pipeline never built is reported, not hidden",
              any("tier" in s for s in app.lb.stale_for()),
              str(app.lb.stale_for()))
        await pilot.press("h")
        await pilot.pause()
        check("back to the built size clears the warning",
              app.lb.stale_for() == [], str(app.lb.stale_for()))

        # useful_max is what stops the late-QB stacking; it must be live.
        for _ in range(30):
            if app.screen.current_label() == "Max useful QB":
                break
            await pilot.press("j")
            await pilot.pause()
        check("j/k skip group headers and reach every field",
              app.screen.current_label() == "Max useful QB",
              str(app.screen.current_label()))
        before = app.lb.useful["QB"]
        await pilot.press("l")
        await pilot.pause()
        check("live-assistant caps apply immediately",
              app.lb.useful["QB"] == before + 1,
              f"{app.lb.useful['QB']} vs {before}")
        await pilot.press("h")
        await pilot.press("escape")
        await pilot.pause()
        check("escape returns to the board",
              not isinstance(app.screen, T.ConfigScreen))
        check("config round-tripped back to where it started",
              app.lb.slot == slot_before
              and app.lb.n_teams == lb_teams_before
              and app.lb.useful["QB"] == before,
              f"slot={app.lb.slot} teams={app.lb.n_teams}")

        print("\n=== persistence ===")
        # The QB fill above went straight into DraftState; every real keypress
        # saves, so mirror that before asserting the round-trip.
        app._save_state()
        check("state file is written", T.STATE_PATH.exists())
        n_before = len(app.st.order)
        mine_before = {str(board.player_name[i]) for i in app.st.mine}

    app2 = T.DraftApp(load_config())
    check("a fresh app restores the saved draft",
          len(app2.st.order) == n_before,
          f"{len(app2.st.order)} vs {n_before}")
    check("it restores which picks were yours",
          {str(app2.board.player_name[i]) for i in app2.st.mine} == mine_before)

    app3 = T.DraftApp(load_config(), reset=True)
    check("--reset starts clean", len(app3.st.order) == 0)

    await reload_checks()
    await sources_checks()
    await sleeper_checks()


async def sources_checks() -> None:
    """Provenance must be visible and re-pullable from inside the dashboard."""
    print("\n=== sources / provenance ===")
    cfg = load_config()
    if T.STATE_PATH.exists():
        T.STATE_PATH.unlink()
    app = T.DraftApp(cfg, reset=True)
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause()

        rows = T.manifest_rows()
        check("every pulled dataset is accounted for", len(rows) >= 15,
              f"{len(rows)} entries")
        check("each row carries a source URL",
              all(r["url"] for r in rows),
              str([r["key"] for r in rows if not r["url"]]))
        check("each row carries an age",
              all(r["age_days"] is not None for r in rows),
              str([r["key"] for r in rows if r["age_days"] is None]))
        check("the market feeds are grouped so refresh knows what to re-pull",
              {r["group"] for r in rows if r["key"].startswith("proj_")} == {"m"},
              str({r["key"]: r["group"] for r in rows
                   if r["key"].startswith("proj_")}))
        check("age labels read naturally",
              T._age_label(0.0005) .endswith("m ago")
              and T._age_label(0.25).endswith("h ago")
              and T._age_label(4.0) == "4.0d ago",
              f"{T._age_label(0.0005)} / {T._age_label(0.25)} / {T._age_label(4.0)}")

        await pilot.press("S")
        await pilot.pause()
        check("S opens sources", isinstance(app.screen, T.SourcesScreen),
              type(app.screen).__name__)
        table = app.screen.query_one("#src")
        check("sources table lists the datasets", table.row_count == len(rows),
              f"{table.row_count} rows vs {len(rows)}")

        # Refresh from inside the screen must dispatch the right pipeline call.
        calls = []
        real = app.run_fetch

        async def spy(argv):
            calls.append(argv)
        app.run_fetch = spy
        try:
            await pilot.press("m")
            await pilot.pause()
            check("m from sources re-pulls the market feeds",
                  calls and calls[-1] == ["run_pipeline.py", "--refresh", "market"],
                  str(calls))
            await pilot.press("s")
            await pilot.pause()
            check("s from sources re-pulls the statistics",
                  calls[-1] == ["run_pipeline.py", "--refresh", "nflverse"],
                  str(calls[-1]))
            await pilot.press("r")
            await pilot.pause()
            check("r from sources re-pulls everything",
                  calls[-1] == ["run_pipeline.py"], str(calls[-1]))
        finally:
            app.run_fetch = real
        await pilot.press("escape")
        await pilot.pause()
        check("escape closes sources",
              not isinstance(app.screen, T.SourcesScreen))

        # Staleness: fresh data must not nag, old market data must.
        check("fresh market data raises no warning", app.stale_sources() == "",
              app.stale_sources())
        real_rows = T.manifest_rows
        T.manifest_rows = lambda: [
            {"key": "adp", "rows": 1, "pulled": "", "age_days": 9.0,
             "url": "u", "group": "m"}]
        try:
            msg = app.stale_sources()
            check("stale ADP is called out by name", "adp" in msg, msg)
            check("and surfaced on the main status line",
                  "press S for sources" in app.scarcity_text(),
                  app.scarcity_text().splitlines()[-1])
        finally:
            T.manifest_rows = real_rows
    if T.STATE_PATH.exists():
        T.STATE_PATH.unlink()


async def reload_checks() -> None:
    """Reloading must pick up fresh data WITHOUT losing or reassigning picks."""
    print("\n=== reload from the datasets ===")
    cfg = load_config()
    if T.STATE_PATH.exists():
        T.STATE_PATH.unlink()
    app = T.DraftApp(cfg, reset=True)
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        board = app.board

        check("the news pass reaches the dashboard",
              int((board["notes"].astype(str) != "").sum()) > 20,
              f"{int((board['notes'].astype(str) != '').sum())} notes")
        flagged = board[board["news_flag"].astype(str) != ""]
        check("news flags come through too", len(flagged) > 20,
              f"{len(flagged)} flagged")

        # Mark some players, then reload and prove they are the SAME players.
        for _ in range(3):
            await pilot.press("d", "j")
        await pilot.press("m")
        await pilot.pause()
        before = [(str(board.player_name[p.idx]), p.mine) for p in app.st.order]
        check("picks recorded before reload", len(before) == 4, str(before))

        msg = app.reload_data()
        after = [(str(app.board.player_name[p.idx]), p.mine)
                 for p in app.st.order]
        check("reload keeps every pick, still pointing at the same players",
              after == before, f"{after} vs {before}")
        check("reload reports what it did", "reloaded" in msg, msg)

        # The real hazard: a reload that reorders the board must remap by NAME.
        # Simulate it by reversing the frame behind the app's back.
        real_load = T.load_board
        T.load_board = lambda c: real_load(c).iloc[::-1].reset_index(drop=True)
        try:
            app.reload_data()
        finally:
            T.load_board = real_load
        after2 = [(str(app.board.player_name[p.idx]), p.mine)
                  for p in app.st.order]
        check("picks survive the board being reordered underneath them",
              after2 == before, f"{after2} vs {before}")
        app.reload_data()

        # Freshness watcher.
        app.stamp = (0.0,) * len(T.SOURCES)
        app.check_sources()
        check("a change on disk is noticed", app.stale)
        check("and it tells you to reload", "press R" in app.scarcity_text(),
              app.scarcity_text().splitlines()[-1])
        await pilot.press("R")
        await pilot.pause()
        check("R clears the stale flag", not app.stale)

        # Guard the trap: run_pipeline drops every stage before the one named,
        # and ingest is first - so any re-pull mode that passes --stage would
        # silently skip the download and recompute from cache instead.
        modes = {k: (label, argv) for k, label, _d, argv in T.FETCH_MODES}
        check("every dataset group is reachable from the fetch menu",
              set(modes) == {"n", "m", "s", "a"}, str(sorted(modes)))
        for key in ("m", "s", "a"):
            label, argv = modes[key]
            check(f"fetch '{key}' ({label}) actually re-pulls, not just recompute",
                  "--stage" not in argv, f"argv={argv}")
        check("market mode narrows to the market sources",
              modes["m"][1] == ["run_pipeline.py", "--refresh", "market"],
              str(modes["m"][1]))
        check("statistics mode narrows to the nflverse sources",
              modes["s"][1] == ["run_pipeline.py", "--refresh", "nflverse"],
              str(modes["s"][1]))

        # Fetch menu + a stubbed fetch that fails must not break anything.
        await pilot.press("F")
        await pilot.pause()
        check("F opens the fetch menu", isinstance(app.screen, T.FetchScreen),
              type(app.screen).__name__)
        await pilot.press("escape")
        await pilot.pause()
        check("escape cancels the fetch",
              not isinstance(app.screen, T.FetchScreen))

        keep = len(app.st.order)
        await app.run_fetch(["-c", "import sys; sys.exit(3)"])
        await pilot.pause()
        check("a failed fetch is reported, not swallowed",
              "failed" in app.fetch_msg, app.fetch_msg)
        check("a failed fetch keeps the draft", len(app.st.order) == keep)
        check("a failed fetch clears the busy flag", app.fetching == "")

        await app.run_fetch(["-c", "print('ok')"])
        await pilot.pause()
        check("a successful fetch reloads the board",
              "reloaded" in app.fetch_msg, app.fetch_msg)
        check("and still keeps the draft", len(app.st.order) == keep)
    if T.STATE_PATH.exists():
        T.STATE_PATH.unlink()


async def sleeper_checks() -> None:
    """Drive the Sleeper poller against a stubbed feed, so this does not need a
    live draft and still covers mapping, ownership and failure."""
    import sleeper_sync

    print("\n=== sleeper mirror ===")
    cfg = load_config()
    if T.STATE_PATH.exists():
        T.STATE_PATH.unlink()
    app = T.DraftApp(cfg, slot=5, reset=True, sleeper="123456789",
                     interval=3600)
    board = app.board
    names = [str(n) for n in board.player_name[:6]]
    feed: list[dict] = []
    real = sleeper_sync.fetch_picks
    sleeper_sync.fetch_picks = lambda _id: list(feed)
    try:
        async with app.run_test(size=(160, 48)) as pilot:
            await pilot.pause()
            for n, nm in enumerate(names, start=1):
                first, _, last = nm.partition(" ")
                feed.append({"pick_no": n, "round": 1, "draft_slot": n,
                             "player_id": f"stub{n}",
                             "metadata": {"first_name": first,
                                          "last_name": last}})
            await app.poll_sleeper()
            await pilot.pause()
            check("picks stream in from the feed",
                  len(app.st.order) == len(names), str(len(app.st.order)))
            check("they map onto the right board players",
                  [str(board.player_name[p.idx]) for p in app.st.order] == names)
            check("the pick at your slot is marked as yours",
                  [str(board.player_name[i]) for i in app.st.mine] == [names[4]],
                  str([str(board.player_name[i]) for i in app.st.mine]))
            check("status line shows the connection",
                  "sleeper" in app.sleeper_status, app.sleeper_status)

            # Sleeper is authoritative: an undo on their side must propagate.
            feed.pop()
            await app.poll_sleeper()
            await pilot.pause()
            check("a pick removed upstream disappears locally",
                  len(app.st.order) == len(names) - 1)

            # An unmatched player must be reported, never silently dropped.
            feed.append({"pick_no": 6, "round": 1, "draft_slot": 6,
                         "player_id": "nobody",
                         "metadata": {"first_name": "Fake",
                                      "last_name": "Person"}})
            await app.poll_sleeper()
            await pilot.pause()
            check("an unmatched pick is reported",
                  len(app.sleeper_unmatched) == 1, str(app.sleeper_unmatched))
            check("unmatched picks are flagged in the status",
                  "unmatched" in app.sleeper_status, app.sleeper_status)

            def boom(_id):
                raise RuntimeError("network down")
            sleeper_sync.fetch_picks = boom
            keep = len(app.st.order)
            await app.poll_sleeper()
            await pilot.pause()
            check("a failed poll does not crash the app",
                  app.is_running)
            check("a failed poll keeps the draft state",
                  len(app.st.order) == keep)
            check("a failed poll says so", "offline" in app.sleeper_status,
                  app.sleeper_status)
    finally:
        sleeper_sync.fetch_picks = real
        if T.STATE_PATH.exists():
            T.STATE_PATH.unlink()


def main() -> int:
    asyncio.run(run())
    print(f"\n{PASSES} passed, {len(FAILS)} failed")
    for f in FAILS:
        print(f"  - {f}")
    if T.STATE_PATH.exists():
        T.STATE_PATH.unlink()
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
