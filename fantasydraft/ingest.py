"""Layer 1 - data ingestion.

Pulls every raw dataset the pipeline needs into data/raw/ and records the
source URL + pull timestamp for each one in data/raw/_manifest.json.

nflverse data is read straight from the nflverse-data GitHub release assets.
We deliberately do not use `nfl_data_py`: it is unmaintained and pins
pandas<2 / numpy<2, which conflicts with everything else here. The release
URLs are the same ones that package wraps.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time

import pandas as pd
import requests

from common import (
    DATA_RAW,
    ESPN_POS_ID,
    ESPN_TEAM_ID,
    ensure_dirs,
    load_config,
    norm_pos,
    norm_team,
    record_pull,
)

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
FFOPP = "https://github.com/ffverse/ffopportunity/releases/download/latest-data"
DP_IDS = "https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv"
FFC_ADP = "https://fantasyfootballcalculator.com/api/v1/adp/ppr"
SLEEPER_PROJ = "https://api.sleeper.app/projections/nfl/{season}"
ESPN_URL = (
    "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}"
    "/segments/0/leaguedefaults/3?view=kona_player_info"
)
SHARKS_PAGE = "https://www.fantasysharks.com/apps/bert/forecasts/projections.php"

UA = {"User-Agent": "Mozilla/5.0 (compatible; fantasy-draft-system/1.0)"}
SESSION = requests.Session()
SESSION.headers.update(UA)


def _get(url: str, *, timeout: int = 90, **kw) -> requests.Response:
    last = None
    for attempt in range(3):
        try:
            r = SESSION.get(url, timeout=timeout, **kw)
            r.raise_for_status()
            return r
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def _save(df: pd.DataFrame, key: str, url: str) -> pd.DataFrame:
    path = DATA_RAW / f"{key}.parquet"
    df.to_parquet(path, index=False)
    record_pull(key, url, path, {"rows": len(df), "cols": len(df.columns)})
    print(f"  [ok] {key:28s} {len(df):>6,} rows")
    return df


def _csv(url: str) -> pd.DataFrame:
    r = _get(url)
    # compression cannot be inferred from an in-memory buffer, so pass it
    # explicitly based on the URL suffix.
    comp = "gzip" if url.split("?")[0].endswith(".gz") else "infer"
    return pd.read_csv(io.BytesIO(r.content), low_memory=False, compression=comp)


def _optional(fn, label: str):
    """Run a fetch that is allowed to fail (e.g. a file that does not exist
    yet this early in the calendar) without killing the whole run."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] {label}: {exc}")
        return None


# ---------------------------------------------------------------------------
# nflverse
# ---------------------------------------------------------------------------
def pull_players() -> pd.DataFrame:
    url = f"{NFLVERSE}/players/players.csv"
    return _save(_csv(url), "players", url)


def pull_rosters(season: int) -> pd.DataFrame:
    url = f"{NFLVERSE}/rosters/roster_{season}.csv"
    return _save(_csv(url), "rosters", url)


def pull_depth_charts(season: int) -> pd.DataFrame:
    """Depth charts are published as an append-only log of snapshots. We keep
    only the most recent snapshot per team, which is the current chart."""
    url = f"{NFLVERSE}/depth_charts/depth_charts_{season}.csv"
    df = _csv(url)
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce", utc=True)
    latest = df.groupby("team")["dt"].transform("max")
    df = df[df["dt"] == latest].copy()
    return _save(df, "depth_charts", url)


def pull_injuries(season: int) -> pd.DataFrame | None:
    """Weekly game-status reports. Does not exist until the season starts, so
    this is expected to be missing on an early-summer run."""
    url = f"{NFLVERSE}/injuries/injuries_{season}.csv"
    return _optional(lambda: _save(_csv(url), "injuries", url), "injuries (not published yet)")


def pull_snap_counts(seasons: list[int]) -> pd.DataFrame:
    frames, urls = [], []
    for s in seasons:
        url = f"{NFLVERSE}/snap_counts/snap_counts_{s}.csv"
        try:
            frames.append(_csv(url))
            urls.append(url)
        except Exception:  # noqa: BLE001
            print(f"  [skip] snap_counts_{s}")
    return _save(pd.concat(frames, ignore_index=True), "snap_counts", "; ".join(urls))


def pull_stats_player(seasons: list[int]) -> pd.DataFrame:
    frames, urls = [], []
    for s in seasons:
        url = f"{NFLVERSE}/stats_player/stats_player_reg_{s}.csv"
        try:
            frames.append(_csv(url))
            urls.append(url)
        except Exception:  # noqa: BLE001
            print(f"  [skip] stats_player_reg_{s}")
    return _save(pd.concat(frames, ignore_index=True), "stats_player_season", "; ".join(urls))


def pull_stats_player_week(season: int) -> pd.DataFrame:
    url = f"{NFLVERSE}/stats_player/stats_player_week_{season}.csv"
    return _save(_csv(url), "stats_player_week", url)


def pull_stats_team(seasons: list[int]) -> pd.DataFrame:
    frames, urls = [], []
    for s in seasons:
        url = f"{NFLVERSE}/stats_team/stats_team_reg_{s}.csv"
        try:
            frames.append(_csv(url))
            urls.append(url)
        except Exception:  # noqa: BLE001
            print(f"  [skip] stats_team_reg_{s}")
    return _save(pd.concat(frames, ignore_index=True), "stats_team", "; ".join(urls))


def pull_ngs() -> pd.DataFrame:
    frames, urls = [], []
    for kind in ("receiving", "rushing", "passing"):
        url = f"{NFLVERSE}/nextgen_stats/ngs_{kind}.csv.gz"
        try:
            d = _csv(url)
            d["ngs_kind"] = kind
            frames.append(d)
            urls.append(url)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] ngs_{kind}: {exc}")
    if not frames:
        print("  [warn] no NGS data pulled")
        return pd.DataFrame()
    return _save(pd.concat(frames, ignore_index=True), "ngs", "; ".join(urls))


def pull_schedules() -> pd.DataFrame:
    url = f"{NFLVERSE}/schedules/games.csv"
    return _save(_csv(url), "schedules", url)


def pull_pbp(seasons: list[int]) -> pd.DataFrame:
    """Play-by-play, trimmed to the columns needed for PROE, red-zone and
    goal-line share. Full pbp is ~20MB/season; we keep only what we use."""
    keep = [
        "season", "posteam", "defteam", "play_type", "pass", "rush", "down",
        "ydstogo", "yardline_100", "half_seconds_remaining", "score_differential",
        "wp", "receiver_player_id", "rusher_player_id", "air_yards",
        "pass_attempt", "rush_attempt", "week", "game_id",
    ]
    frames, urls = [], []
    for s in seasons:
        url = f"{NFLVERSE}/pbp/play_by_play_{s}.parquet"
        try:
            r = _get(url, timeout=180)
            d = pd.read_parquet(io.BytesIO(r.content))
            d = d[[c for c in keep if c in d.columns]].copy()
            d["season"] = s
            frames.append(d)
            urls.append(url)
            print(f"  [ok] pbp {s} {len(d):,} plays")
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] pbp_{s}: {exc}")
    return _save(pd.concat(frames, ignore_index=True), "pbp", "; ".join(urls))


def pull_ffopportunity(seasons: list[int]) -> pd.DataFrame | None:
    """Pre-built expected fantasy points (XGBoost on play-by-play) from the
    ffverse sister project. Used as an opportunity-quality feature rather than
    rebuilding an expected-points model."""
    frames, urls = [], []
    for s in seasons:
        url = f"{FFOPP}/ep_weekly_{s}.csv"
        try:
            frames.append(_csv(url))
            urls.append(url)
        except Exception:  # noqa: BLE001
            print(f"  [skip] ffopportunity {s}")
    if not frames:
        return None
    return _save(pd.concat(frames, ignore_index=True), "ffopportunity", "; ".join(urls))


def pull_id_crosswalk() -> pd.DataFrame:
    """DynastyProcess maintains the cross-site player ID map (sleeper / espn /
    gsis / fantasypros / pfr in one table). This is the join hub."""
    return _save(_csv(DP_IDS), "id_crosswalk", DP_IDS)


# ---------------------------------------------------------------------------
# ADP + projections
# ---------------------------------------------------------------------------
def pull_adp(season: int, team_counts: list[int]) -> pd.DataFrame:
    frames, urls = [], []
    for teams in team_counts:
        url = f"{FFC_ADP}?teams={teams}&year={season}&position=all"
        payload = _get(url).json()
        players = payload.get("players") or []
        if not players:
            print(f"  [skip] adp {teams}-team: no players returned")
            continue
        d = pd.DataFrame(players)
        d["teams"] = teams
        d["adp_window_start"] = payload.get("meta", {}).get("start_date")
        d["adp_window_end"] = payload.get("meta", {}).get("end_date")
        d["total_drafts"] = payload.get("meta", {}).get("total_drafts")
        frames.append(d)
        urls.append(url)
    return _save(pd.concat(frames, ignore_index=True), "adp", "; ".join(urls))


def pull_sleeper(season: int) -> pd.DataFrame:
    rows, urls = [], []
    for pos in ("QB", "RB", "WR", "TE", "K", "DEF"):
        url = SLEEPER_PROJ.format(season=season) + (
            f"?season_type=regular&position[]={pos}&order_by=pts_ppr"
        )
        try:
            data = _get(url).json()
        except Exception:  # noqa: BLE001
            print(f"  [skip] sleeper {pos}")
            continue
        urls.append(url)
        for rec in data:
            stats = rec.get("stats") or {}
            pts = stats.get("pts_ppr")
            if pts is None:
                pts = stats.get("pts_std")
            if not pts:
                continue
            p = rec.get("player") or {}
            rows.append({
                "sleeper_id": rec.get("player_id"),
                "player_name": f"{p.get('first_name','')} {p.get('last_name','')}".strip(),
                "position": norm_pos(p.get("position") or pos),
                "team": norm_team(rec.get("team") or p.get("team")),
                "proj_points": float(pts),
                "games": stats.get("gp"),
                "adp_ppr_sleeper": stats.get("adp_ppr"),
                "injury_status": p.get("injury_status"),
                "years_exp": p.get("years_exp"),
            })
    return _save(pd.DataFrame(rows), "proj_sleeper", "; ".join(urls))


def pull_espn(season: int) -> pd.DataFrame:
    url = ESPN_URL.format(season=season)
    hdr = {
        "x-fantasy-filter": json.dumps({
            "players": {
                "limit": 900,
                "sortDraftRanks": {
                    "sortPriority": 1, "sortAsc": True, "value": "PPR",
                },
            }
        })
    }
    data = _get(url, headers=hdr).json()
    rows = []
    for entry in data.get("players", []):
        p = entry.get("player") or {}
        total = None
        for s in p.get("stats", []):
            if (s.get("seasonId") == season and s.get("statSourceId") == 1
                    and s.get("statSplitTypeId") == 0):
                total = s.get("appliedTotal")
                break
        if not total:
            continue
        ranks = (p.get("draftRanksByRankType") or {}).get("PPR") or {}
        rows.append({
            "espn_id": p.get("id"),
            "player_name": p.get("fullName"),
            "position": ESPN_POS_ID.get(p.get("defaultPositionId")),
            "team": ESPN_TEAM_ID.get(p.get("proTeamId")),
            "proj_points": float(total),
            "espn_rank": ranks.get("rank"),
            "espn_auction": ranks.get("auctionValue"),
            "injured": p.get("injured"),
            "injury_status": p.get("injuryStatus"),
        })
    df = pd.DataFrame(rows)
    df = df[df["position"].notna()]
    return _save(df, "proj_espn", url)


def _sharks_segment(season: int) -> int:
    """The FantasySharks season segment id changes every year, so read it off
    the page's dropdown instead of hardcoding."""
    html = _get(SHARKS_PAGE).text
    m = re.search(r"name=[\"']?Segment.*?</select>", html, re.S | re.I)
    if m:
        for val, label in re.findall(r"value=[\"']?(\d+)[\"']?[^>]*>([^<]+)", m.group(0)):
            if re.search(rf"{season}\s+NFL\s+Season", label):
                return int(val)
    raise RuntimeError(f"could not find FantasySharks segment for {season}")


def pull_sharks(season: int) -> pd.DataFrame:
    segment = _sharks_segment(season)
    url = (f"{SHARKS_PAGE}?League=-1&Position=99&scoring=2"
           f"&Segment={segment}&uid=4&csv=1")
    df = _csv(url)
    df = df.rename(columns={"Player Name": "player_name", "Team": "team",
                            "Position": "position", "Pts": "proj_points"})
    df["position"] = df["position"].map(norm_pos)
    df["team"] = df["team"].map(norm_team)
    df = df[df["position"].isin(["QB", "RB", "WR", "TE", "K", "DEF"])]
    df = df[["player_name", "team", "position", "proj_points"]].dropna(
        subset=["proj_points"])
    return _save(df, "proj_fantasysharks", url)


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Layer 1 - pull all raw datasets")
    ap.add_argument("--season", type=int, default=cfg["season"])
    ap.add_argument("--history", type=int, default=3,
                    help="how many completed seasons of history to pull")
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset: market | nflverse | all")
    args = ap.parse_args(argv)

    ensure_dirs()
    season = args.season
    hist = [season - i for i in range(1, args.history + 1)]
    scope = set(args.only or ["all"])
    do_nfl = {"all", "nflverse"} & scope
    do_mkt = {"all", "market"} & scope

    print(f"\n=== Layer 1: ingest (season {season}, history {hist}) ===")

    if do_nfl:
        print("- nflverse core")
        pull_players()
        pull_id_crosswalk()
        pull_rosters(season)
        pull_depth_charts(season)
        pull_injuries(season)
        pull_schedules()
        print("- nflverse stats")
        pull_stats_player(hist)
        pull_stats_player_week(hist[0])
        pull_stats_team(hist)
        pull_snap_counts(hist)
        pull_ngs()
        print("- play-by-play (PROE / red zone)")
        pull_pbp(hist)
        print("- expected fantasy points")
        pull_ffopportunity(hist)

    if do_mkt:
        print("- market: ADP + projections")
        pull_adp(season, cfg["league"]["team_counts"])
        # One projection feed going down must not take the pipeline with it.
        # These are third-party sites with no contract to us - FantasySharks in
        # particular 403s whole IP ranges - and the blend already renormalizes
        # over whichever sources answered. A stale cached copy from the last
        # run is far better than no board at all on draft morning.
        alive = 0
        for name, fn in (("sleeper", pull_sleeper), ("espn", pull_espn),
                         ("fantasysharks", pull_sharks)):
            try:
                fn(season)
                alive += 1
            except Exception as exc:  # noqa: BLE001
                cached = (DATA_RAW / f"proj_{name}.parquet").exists()
                print(f"  [warn] {name} projections unavailable ({exc}); "
                      + ("using the cached copy" if cached else "skipping"))
                if cached:
                    alive += 1
        if alive < cfg["projections"]["min_sources"]:
            raise SystemExit(
                f"only {alive} projection source(s) available, need "
                f"{cfg['projections']['min_sources']} - refusing to build a "
                "board on a single source")

    print(f"\nmanifest -> {DATA_RAW / '_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
