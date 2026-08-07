"""Shared paths, name/team normalization, and data-provenance helpers."""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import yaml

PKG_ROOT = Path(__file__).resolve().parent.parent


def _default_home() -> Path:
    """Where the data, config and output live.

    Running from a source checkout keeps everything in the repo, which is what
    the pipeline and every test expect. Once installed as a package there is no
    writable repo, so fall back to the platform's user-data directory.
    `FANTASY_HOME` overrides both - useful for keeping separate leagues apart.
    """
    env = os.environ.get("FANTASY_HOME")
    if env:
        return Path(env).expanduser()
    # A checkout is identifiable by the project file next to the package.
    if (PKG_ROOT / "pyproject.toml").exists() and (PKG_ROOT / "fantasydraft").is_dir():
        return PKG_ROOT
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData"
                                                  / "Local")
        return Path(base) / "FantasyDraftBoard"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "FantasyDraftBoard"
    base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / "fantasy-draftboard"


ROOT = _default_home()
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
CONFIG = ROOT / "config"
OUTPUT = ROOT / "output"
MANIFEST = DATA_RAW / "_manifest.json"
# Shipped defaults, used to seed CONFIG on an installed first run. Installed,
# they sit inside the package; in a checkout they are the repo's config/.
_PKG_CONFIG = Path(__file__).resolve().parent / "config"
BUNDLED_CONFIG = _PKG_CONFIG if _PKG_CONFIG.is_dir() else PKG_ROOT / "config"

FANTASY_POS = ["QB", "RB", "WR", "TE", "K", "DEF"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_config(name: str = "weights.yaml") -> dict:
    with open(CONFIG / name) as fh:
        return yaml.safe_load(fh)


# Flag -> (label, meaning). Colours are defined in the output layer.
NEWS_FLAGS = {
    "out": "downgrade / injury",
    "down": "downgrade",
    "watch": "volatile - watch",
    "up": "upgrade / rising",
}


def load_news(name: str = "news.yaml") -> dict:
    """Manual draft-day player notes, keyed by normalized player name.

    Returns {norm_name: {"note": str, "flag": str|None}}. Missing file is fine -
    the board simply carries no manual notes. This is the channel a draft-day
    agent writes into: edit config/news.yaml, rebuild the output stage."""
    path = CONFIG / name
    if not path.exists():
        return {}
    with open(path) as fh:
        doc = yaml.safe_load(fh) or {}
    out = {}
    for player, entry in (doc.get("players") or {}).items():
        if entry is None:
            continue
        if isinstance(entry, str):
            entry = {"note": entry}
        note = str(entry.get("note", "")).strip()
        flag = entry.get("flag")
        flag = str(flag).strip().lower() if flag else None
        if note or flag:
            out[norm_name(player)] = {"note": note, "flag": flag}
    return out


# --------------------------------------------------------------------------
# Provenance manifest: every raw file records where it came from and when.
# --------------------------------------------------------------------------
def read_manifest() -> dict:
    if MANIFEST.exists():
        with open(MANIFEST) as fh:
            return json.load(fh)
    return {}


def write_manifest(manifest: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


def record_pull(key: str, url: str, path: Path, extra: dict | None = None) -> None:
    manifest = read_manifest()
    manifest[key] = {
        "url": url,
        "path": str(path.relative_to(ROOT)),
        "pulled_at": now_iso(),
        "bytes": path.stat().st_size if path.exists() else 0,
        **(extra or {}),
    }
    write_manifest(manifest)


# --------------------------------------------------------------------------
# Team abbreviations -> nflverse canonical
# --------------------------------------------------------------------------
TEAM_FIXES = {
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
    "GNB": "GB", "GBP": "GB", "KAN": "KC", "KCC": "KC",
    "NWE": "NE", "NEP": "NE", "NOR": "NO", "NOS": "NO",
    "SFO": "SF", "SFN": "SF", "TAM": "TB", "TBB": "TB",
    "LVR": "LV", "OAK": "LV", "RAI": "LV",
    "SD": "LAC", "SDG": "LAC", "LACH": "LAC",
    "STL": "LA", "LAR": "LA", "RAM": "LA",
    "JAC": "JAX", "WSH": "WAS", "WFT": "WAS",
    "NYA": "NYJ", "NYN": "NYG",
}

# ESPN proTeamId -> nflverse abbreviation
ESPN_TEAM_ID = {
    1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN",
    8: "DET", 9: "GB", 10: "TEN", 11: "IND", 12: "KC", 13: "LV", 14: "LA",
    15: "MIA", 16: "MIN", 17: "NE", 18: "NO", 19: "NYG", 20: "NYJ",
    21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC", 25: "SF", 26: "SEA",
    27: "TB", 28: "WAS", 29: "CAR", 30: "JAX", 33: "BAL", 34: "HOU",
}

ESPN_POS_ID = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DEF"}


def norm_team(team) -> str | None:
    if team is None:
        return None
    t = str(team).strip().upper()
    if not t or t in {"NAN", "NONE", "FA", "-"}:
        return None
    return TEAM_FIXES.get(t, t)


# --------------------------------------------------------------------------
# Player name normalization -> merge key
# --------------------------------------------------------------------------
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# Common nickname / spelling divergences between sources.
_NAME_ALIASES = {
    "mitch trubisky": "mitchell trubisky",
    "gabe davis": "gabriel davis",
    "josh palmer": "joshua palmer",
    "cam ward": "cameron ward",
    "chig okonkwo": "chigoziem okonkwo",
    "demarcus robinson": "demarcus robinson",
    "kenneth walker": "kenneth walker iii",
    "marquise brown": "hollywood brown",
    "jeff wilson": "jeffery wilson",
    "chris brooks": "christopher brooks",
    "tank bigsby": "thomas bigsby",
    "deebo samuel": "deebo samuel",
}


def norm_name(name) -> str:
    """Lowercase, strip accents/punctuation/suffixes -> stable merge key."""
    if name is None:
        return ""
    s = str(name)
    if "," in s:  # "Allen, Josh" -> "Josh Allen"
        parts = [p.strip() for p in s.split(",", 1)]
        if len(parts) == 2 and parts[1]:
            s = f"{parts[1]} {parts[0]}"
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"[.'`’]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    tokens = [t for t in s.split() if t not in _SUFFIXES]
    s = " ".join(tokens)
    return _NAME_ALIASES.get(s, s)


def merge_key(name, position=None) -> str:
    """Name key scoped by position bucket, so a WR and a DB sharing a name
    do not collide when joining projection sources."""
    n = norm_name(name)
    if position is None:
        return n
    return f"{n}|{norm_pos(position)}"


def norm_pos(pos) -> str | None:
    if pos is None:
        return None
    p = str(pos).strip().upper()
    if p in {"DST", "D/ST", "D", "DEF", "TEAM"}:
        return "DEF"
    if p in {"PK", "K"}:
        return "K"
    if p in {"FB"}:
        return "RB"
    return p


def ensure_dirs() -> None:
    for d in (DATA_RAW, DATA_PROC, OUTPUT, CONFIG):
        d.mkdir(parents=True, exist_ok=True)
    seed_config()


def seed_config() -> None:
    """Copy the shipped defaults into a fresh user config directory.

    Only fills gaps - an existing weights.yaml or news.yaml is never
    overwritten, so an upgrade cannot silently discard your league setup or a
    draft-day news pass."""
    if BUNDLED_CONFIG.resolve() == CONFIG.resolve():
        return                                   # running from a checkout
    CONFIG.mkdir(parents=True, exist_ok=True)
    for src in BUNDLED_CONFIG.glob("*.yaml"):
        dst = CONFIG / src.name
        if not dst.exists():
            shutil.copy2(src, dst)


# Copies derived from a built board (a live draft mirror, a scratch experiment)
# must never be mistaken for the board itself - verifying or drafting against
# one silently tests stale, already-marked-up state.
DERIVED_MARKERS = ("_live", "_marked", "_scratch")


def latest_board(pattern: str = "draft_board_*.xlsx"):
    """Newest built workbook in output/, ignoring derived copies."""
    hits = [p for p in OUTPUT.glob(pattern)
            if not any(m in p.stem for m in DERIVED_MARKERS)]
    return sorted(hits)[-1] if hits else None
