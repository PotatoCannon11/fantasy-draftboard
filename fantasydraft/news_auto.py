"""Automated draft-day news: buzz + reports, populated for the whole board.

Two free, unauthenticated feeds, refreshed on demand (seconds, no full ingest):

  * Sleeper trending adds/drops  -> BUZZ. What the fantasy crowd is moving on in
    the last ~48h. High adds = rising interest (flag up); high drops = falling
    (flag down). Magnitude is the signal.
  * ESPN NFL news feed           -> REPORTS. Recent headlines matched to players
    by name / tagged athlete, with light keyword sentiment (injury -> down,
    won-job -> up, else neutral watch).

This is Channel 3. It never overrides the two curated channels: a manual note in
config/news.yaml and an auto injury designation both outrank anything here (see
build_spreadsheet._apply_news). The point is to fill the *empty* Notes cells so
you walk into the draft with the crowd's read on every player, then hand-correct
the few that matter.

Cache: data/raw/news_auto.parquet. Refresh: `.venv/bin/python refresh_news.py`.
"""
from __future__ import annotations

import re
import time

import pandas as pd
import requests

from common import DATA_PROC, DATA_RAW, norm_name, record_pull
from idmap import IdResolver

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (fantasy-draft-system)"})

SLEEPER_TREND = "https://api.sleeper.app/v1/players/nfl/trending/{kind}"
SLEEPER_PLAYERS = "https://api.sleeper.app/v1/players/nfl"
ESPN_NEWS = ("https://site.api.espn.com/apis/site/v2/sports/football/nfl/news"
             "?limit=50")

# Keyword sentiment for a report headline/description.
_DOWN = re.compile(r"\b(injur|hurt|out|ir|pup|surger|torn|acl|sprain|strain|"
                   r"concuss|hamstring|suspend|arrest|holdout|hold out|carted|"
                   r"questionable|doubtful|setback|miss(es|ed)?|sidelined)\b", re.I)
_UP = re.compile(r"\b(starter|wr1|rb1|te1|lead back|breakout|returns?|activated|"
                 r"cleared|promoted|first team|locked in|standout|dominat|"
                 r"steal|sleeper|hype|buzz|impress)\b", re.I)


def _get_json(url: str, timeout: int = 30):
    last = None
    for attempt in range(3):
        try:
            r = SESSION.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


# ---------------------------------------------------------------------------
# Channel 3a - Sleeper trending (buzz)
# ---------------------------------------------------------------------------
def _norm_sid(x) -> str | None:
    try:
        return str(int(float(x)))
    except (TypeError, ValueError):
        return str(x) if x is not None else None


def sleeper_buzz(resolver: IdResolver, hours: int = 48, limit: int = 60) -> dict:
    """{uid: (note, flag)} for players trending up (adds) or down (drops)."""
    out: dict[str, tuple[str, str]] = {}
    # resolver.by_sleeper stores float-stringified ids ("96.0"); the trending
    # feed returns plain ids ("11571"). Normalize both sides to integer strings.
    sid_to_uid = {}
    for k, uid in resolver.by_sleeper.items():
        nk = _norm_sid(k)
        if nk:
            sid_to_uid.setdefault(nk, uid)
    for kind, flag, arrow in (("add", "up", "▲"), ("drop", "down", "▼")):
        try:
            rows = _get_json(SLEEPER_TREND.format(kind=kind)
                             + f"?lookback_hours={hours}&limit={limit}")
        except RuntimeError:
            continue
        for row in rows:
            cnt = int(row.get("count", 0))
            if cnt < 300:                      # ignore trickle
                continue
            uid = sid_to_uid.get(_norm_sid(row.get("player_id")))
            if uid is None:
                continue                       # not a player we rank; skip
            k = f"{cnt/1000:.1f}k" if cnt >= 1000 else str(cnt)
            note = f"{arrow} {k} {kind}s/48h (Sleeper)"
            # Adds outrank drops if a player somehow appears in both.
            if uid not in out or flag == "up":
                out[uid] = (note, flag)
    return out


# ---------------------------------------------------------------------------
# Channel 3b - ESPN headlines (reports)
# ---------------------------------------------------------------------------
def espn_reports(name_index: dict) -> dict:
    """name_index: {norm_name: uid}. Returns {uid: (note, flag)} from recent
    ESPN NFL headlines, matched by tagged athlete first, then by name text."""
    try:
        doc = _get_json(ESPN_NEWS)
    except RuntimeError:
        return {}
    out: dict[str, tuple[str, str]] = {}
    for art in doc.get("articles", []):
        head = str(art.get("headline", "")).strip()
        desc = str(art.get("description", "")).strip()
        if not head:
            continue
        # Require the player's NAME to actually appear in the headline/desc, so
        # generic team articles ("Raiders camp preview") tagged with 20 athletes
        # do not spray a useless note across the board. A tagged athlete only
        # counts if the name is also in the text.
        clean = re.sub(r"[^a-z0-9]+", " ", f"{head} {desc}".lower())
        flag = "down" if _DOWN.search(f"{head} {desc}") else (
            "up" if _UP.search(f"{head} {desc}") else "watch")
        note = head if len(head) <= 90 else head[:87] + "..."
        for nkey, uid in name_index.items():
            toks = nkey.split()
            if len(toks) < 2:
                continue
            phrase = f"{toks[0]} {toks[-1]}"
            if phrase in clean or nkey in clean:
                out.setdefault(uid, (note, flag))   # first (most recent) wins
    return out


# ---------------------------------------------------------------------------
def _board_players() -> pd.DataFrame:
    """Skill-position players actually on the board (draftable universe). We do
    not match news against team defenses or O-linemen - their names collide with
    generic team-preview headlines."""
    path = DATA_PROC / "draft_board.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["player_uid", "player_name", "position"])
    b = pd.read_parquet(path)
    return b[b["position"].isin(["QB", "RB", "WR", "TE", "K"])][
        ["player_uid", "player_name", "position"]]


def build(resolver: IdResolver | None = None) -> pd.DataFrame:
    resolver = resolver or IdResolver()
    board = _board_players()
    board_uids = set(board["player_uid"])
    # name -> uid, restricted to on-board skill players.
    name_index = {}
    for uid, nm in zip(board["player_uid"], board["player_name"]):
        if nm:
            name_index.setdefault(norm_name(nm), uid)
    uid_name = dict(zip(board["player_uid"], board["player_name"]))

    buzz = {u: v for u, v in sleeper_buzz(resolver).items() if u in board_uids}
    reports = espn_reports(name_index)
    print(f"  buzz (Sleeper)    {len(buzz):>5} players")
    print(f"  reports (ESPN)    {len(reports):>5} matched")

    rows = []
    for uid in set(buzz) | set(reports):
        bits, flag = [], None
        if uid in reports:
            rnote, rflag = reports[uid]
            bits.append(rnote)
            flag = rflag
        if uid in buzz:
            bnote, bflag = buzz[uid]
            bits.append(bnote)
            flag = flag or bflag           # report sentiment wins if present
        rows.append({
            "player_uid": uid,
            "player_name": uid_name.get(uid, ""),
            "auto_note": " | ".join(bits),
            "auto_flag": flag or "watch",
        })
    return pd.DataFrame(rows, columns=["player_uid", "player_name",
                                       "auto_note", "auto_flag"])


def refresh() -> pd.DataFrame:
    df = build()
    path = DATA_RAW / "news_auto.parquet"
    df.to_parquet(path, index=False)
    record_pull("news_auto", "sleeper-trending + espn-news", path,
                {"rows": len(df)})
    return df


def load_auto_news() -> dict:
    """{norm_name: {'note': str, 'flag': str}} from the cache, or {} if none."""
    path = DATA_RAW / "news_auto.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    out = {}
    for _, r in df.iterrows():
        nm = norm_name(r.get("player_name"))
        if nm:
            out[nm] = {"note": str(r.get("auto_note") or ""),
                       "flag": (str(r.get("auto_flag")) or None) or None}
    return out


if __name__ == "__main__":
    print("=== Auto-news: Sleeper buzz + ESPN reports ===")
    d = refresh()
    print(d.head(20).to_string(index=False))
