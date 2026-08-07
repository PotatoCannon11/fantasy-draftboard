"""Cross-source player identity resolution.

The four market sources (Sleeper, ESPN, FantasySharks, FantasyFootballCalculator)
each use their own player keys, and only some of them overlap. This module
builds one resolver that maps any of them onto a canonical `player_uid`.

Resolution order, most to least reliable:
  1. sleeper_id  -> gsis_id
  2. espn_id     -> gsis_id
  3. normalized name + position  -> gsis_id
  4. fall back to the name+position key itself (unrostered / very new players)

Team defenses have no player id anywhere, so they are keyed as ``DEF:<TEAM>``.
"""
from __future__ import annotations

import pandas as pd

from common import DATA_RAW, merge_key, norm_name, norm_pos, norm_team


def _read(name: str) -> pd.DataFrame:
    path = DATA_RAW / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


class IdResolver:
    """Maps source-specific ids and names onto a canonical player_uid."""

    def __init__(self) -> None:
        self.by_sleeper: dict[str, str] = {}
        self.by_espn: dict[str, str] = {}
        self.by_name: dict[str, str] = {}
        self.info: dict[str, dict] = {}
        self.alias: dict[str, str] = {}
        self._name_groups: dict[str, set[str]] = {}
        self._build()
        self._collapse_aliases()

    # -- construction -----------------------------------------------------
    def _register(self, uid, *, name=None, position=None, team=None,
                  sleeper_id=None, espn_id=None, gsis_id=None,
                  overwrite=False) -> None:
        if not uid:
            return
        rec = self.info.setdefault(uid, {})
        for key, val in (("player_name", name), ("position", norm_pos(position)),
                         ("team", norm_team(team)), ("gsis_id", gsis_id)):
            if val is not None and val == val and (overwrite or not rec.get(key)):
                rec[key] = val
        if sleeper_id and str(sleeper_id) != "nan":
            self.by_sleeper.setdefault(str(sleeper_id), uid)
        if espn_id and str(espn_id) != "nan":
            self.by_espn.setdefault(str(int(float(espn_id))), uid)
        if name:
            nk = merge_key(name, position)
            self._name_groups.setdefault(nk, set()).add(uid)
            if overwrite or nk not in self.by_name:
                self.by_name[nk] = uid

    def _build(self) -> None:
        # 1. Current-season rosters: the most trustworthy identity source for
        #    players who are actually on a team right now.
        rosters = _read("rosters")
        if not rosters.empty:
            for r in rosters.itertuples(index=False):
                gsis = getattr(r, "gsis_id", None)
                if not gsis or str(gsis) == "nan":
                    continue
                self._register(
                    gsis, name=getattr(r, "full_name", None),
                    position=getattr(r, "position", None),
                    team=getattr(r, "team", None),
                    sleeper_id=getattr(r, "sleeper_id", None),
                    espn_id=getattr(r, "espn_id", None),
                    gsis_id=gsis, overwrite=True,
                )

        # 2. DynastyProcess crosswalk: fills in free agents, rookies and anyone
        #    the roster snapshot has not caught up to.
        xw = _read("id_crosswalk")
        if not xw.empty:
            for r in xw.itertuples(index=False):
                gsis = getattr(r, "gsis_id", None)
                name = getattr(r, "name", None)
                pos = getattr(r, "position", None)
                uid = gsis if gsis and str(gsis) != "nan" else merge_key(name, pos)
                self._register(
                    uid, name=name, position=pos, team=getattr(r, "team", None),
                    sleeper_id=getattr(r, "sleeper_id", None),
                    espn_id=getattr(r, "espn_id", None),
                    gsis_id=gsis if gsis and str(gsis) != "nan" else None,
                )

        # 3. nflverse players table: last-resort name coverage.
        players = _read("players")
        if not players.empty:
            for r in players.itertuples(index=False):
                gsis = getattr(r, "gsis_id", None)
                if not gsis or str(gsis) == "nan":
                    continue
                self._register(
                    gsis, name=getattr(r, "display_name", None),
                    position=getattr(r, "position", None),
                    team=getattr(r, "latest_team", None),
                    espn_id=getattr(r, "espn_id", None), gsis_id=gsis,
                )

    def _collapse_aliases(self) -> None:
        """Sources disagree on player ids for rookies: the DynastyProcess
        crosswalk ships placeholder gsis ids (e.g. ``LOV121782``) until the
        real nflverse id (``00-00xxxxx``) is issued, so the same rookie can end
        up registered twice. Collapse uids that share a name+position key onto
        one canonical id, preferring a real gsis id."""
        def rank(uid: str) -> tuple:
            return (
                0 if uid.startswith("00-") else 1,   # real gsis id wins
                0 if not uid.startswith(("NAME:", "DEF:")) else 1,
                -len(self.info.get(uid, {})),        # richest record wins
                uid,
            )

        for nk, uids in self._name_groups.items():
            if len(uids) < 2:
                continue
            canonical = sorted(uids, key=rank)[0]
            merged = dict(self.info.get(canonical, {}))
            for uid in uids:
                if uid == canonical:
                    continue
                self.alias[uid] = canonical
                for k, v in self.info.get(uid, {}).items():
                    merged.setdefault(k, v)
            self.info[canonical] = merged
            self.by_name[nk] = canonical

        # Re-point the id indexes at canonical uids.
        for index in (self.by_sleeper, self.by_espn, self.by_name):
            for key, uid in list(index.items()):
                if uid in self.alias:
                    index[key] = self.alias[uid]

    def canonical(self, uid: str | None) -> str | None:
        seen = set()
        while uid in self.alias and uid not in seen:
            seen.add(uid)
            uid = self.alias[uid]
        return uid

    # -- lookup -----------------------------------------------------------
    def resolve(self, **kw) -> str | None:
        return self.canonical(self._resolve_raw(**kw))

    def _resolve_raw(self, *, sleeper_id=None, espn_id=None, name=None,
                     position=None, team=None) -> str | None:
        pos = norm_pos(position)
        if pos == "DEF":
            t = norm_team(team) or norm_team(_def_team_from_name(name))
            return f"DEF:{t}" if t else None

        if sleeper_id is not None and str(sleeper_id) != "nan":
            uid = self.by_sleeper.get(str(sleeper_id))
            if uid:
                return uid
        if espn_id is not None and str(espn_id) != "nan":
            try:
                uid = self.by_espn.get(str(int(float(espn_id))))
            except (TypeError, ValueError):
                uid = None
            if uid:
                return uid
        if name:
            uid = self.by_name.get(merge_key(name, pos))
            if uid:
                return uid
            # Position-agnostic retry: sources disagree on RB/FB, WR/TE etc.
            nk = norm_name(name)
            for cand_pos in ("QB", "RB", "WR", "TE", "K"):
                uid = self.by_name.get(f"{nk}|{cand_pos}")
                if uid:
                    return uid
            # Unknown player: key on the name itself so the row still survives.
            return f"NAME:{nk}|{pos}"
        return None

    def attrs(self, uid: str) -> dict:
        uid = self.canonical(uid)
        if uid and uid.startswith("DEF:"):
            return {"player_name": f"{uid[4:]} DEF", "position": "DEF",
                    "team": uid[4:], "gsis_id": None}
        return self.info.get(uid, {})


# Team-defense names arrive as "Chiefs D/ST", "Kansas City Chiefs", "KCC" etc.
_NICK_TO_TEAM = {
    "cardinals": "ARI", "falcons": "ATL", "ravens": "BAL", "bills": "BUF",
    "panthers": "CAR", "bears": "CHI", "bengals": "CIN", "browns": "CLE",
    "cowboys": "DAL", "broncos": "DEN", "lions": "DET", "packers": "GB",
    "texans": "HOU", "colts": "IND", "jaguars": "JAX", "chiefs": "KC",
    "raiders": "LV", "chargers": "LAC", "rams": "LA", "dolphins": "MIA",
    "vikings": "MIN", "patriots": "NE", "saints": "NO", "giants": "NYG",
    "jets": "NYJ", "eagles": "PHI", "steelers": "PIT", "seahawks": "SEA",
    "49ers": "SF", "niners": "SF", "buccaneers": "TB", "titans": "TEN",
    "commanders": "WAS", "football": "WAS", "redskins": "WAS",
}


def _def_team_from_name(name) -> str | None:
    if not name:
        return None
    tokens = norm_name(name).split()
    for tok in tokens:
        if tok in _NICK_TO_TEAM:
            return _NICK_TO_TEAM[tok]
    for tok in tokens:
        t = norm_team(tok)
        if t and len(t) <= 3:
            return t
    return None


def resolve_frame(df: pd.DataFrame, resolver: IdResolver, *,
                  sleeper_col=None, espn_col=None, name_col="player_name",
                  pos_col="position", team_col="team") -> pd.DataFrame:
    """Add a `player_uid` column to a source frame."""
    out = df.copy()
    uids = []
    for r in out.itertuples(index=False):
        uids.append(resolver.resolve(
            sleeper_id=getattr(r, sleeper_col, None) if sleeper_col else None,
            espn_id=getattr(r, espn_col, None) if espn_col else None,
            name=getattr(r, name_col, None) if name_col else None,
            position=getattr(r, pos_col, None) if pos_col else None,
            team=getattr(r, team_col, None) if team_col else None,
        ))
    out["player_uid"] = uids
    return out
