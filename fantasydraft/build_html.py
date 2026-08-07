"""Renders the draft board as a self-contained HTML view.

Same data as the xlsx, but operable in a browser: live scarcity, tier-cliff
separators sized by the actual VBD drop, and click-to-mark-drafted. Written as
one file with the data inlined so it works offline and can be opened from
anywhere.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date

import numpy as np
import pandas as pd

from common import (
    DATA_PROC,
    FANTASY_POS,
    OUTPUT,
    ensure_dirs,
    load_config,
    load_news,
    norm_name,
    read_manifest,
)
from news_auto import load_auto_news
from tiering import replacement_ranks

FIELDS = [
    "overall_rank", "player_name", "position", "team", "tier", "tier_label",
    "sub_tier_label", "final_projection", "vbd_score", "vbd_8", "vbd_10",
    "projection_low", "projection_high", "adp_10", "adp_8", "value_rounds",
    "bye_week", "sos_z", "age", "games_missed_l3y", "n_sources",
    "context_multiplier", "depth_rank", "note", "flag",
]

_INJ_DOWN = {"IR", "PUP", "NFI", "SUSP", "OUT", "DOUBTFUL", "DNR"}
_INJ_WATCH = {"QUESTIONABLE"}


def _news_for(row, news: dict, auto: dict) -> tuple[str, str]:
    """Combine three channels: auto injury designation, manual news.yaml note,
    and automated buzz/reports. Curated channels win; automated fills gaps."""
    inj = str(row.get("injury_status") or "").strip().upper()
    key = norm_name(row.get("player_name"))
    manual = news.get(key, {})
    bits, flag = [], manual.get("flag")
    if inj:
        bits.append(inj)
        if not flag:
            flag = "out" if inj in _INJ_DOWN else ("watch" if inj in _INJ_WATCH else "")
    if manual.get("note"):
        bits.append(manual["note"])
    if not bits:
        auto_n = auto.get(key, {})
        if auto_n.get("note"):
            bits.append(auto_n["note"])
            if not flag:
                flag = auto_n.get("flag") or ""
    return " - ".join(bits), (flag or "")


def _clean(v):
    """JSON-safe scalar. Note `iterrows` hands back plain Python ints, and
    `bool` subclasses `int`, so both need checking before the int branch."""
    if v is None:
        return None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 3)
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v)


def build(cfg: dict, out_path=None) -> str:
    ensure_dirs()
    path = DATA_PROC / "draft_board.parquet"
    if not path.exists():
        raise SystemExit("draft_board.parquet missing - run board.py first")
    df = pd.read_parquet(path)

    df = df.sort_values("vbd_score", ascending=False).reset_index(drop=True)
    ocfg = cfg["output"]
    keep = (df["n_sources"].fillna(0) >= 2) | (
        df["vbd_score"].rank(ascending=False, method="min")
        <= ocfg.get("single_source_rank_grace", 60))
    if ocfg.get("single_source_keep_if_adp", True):
        keep |= df[f"adp_{ocfg['adp_teams']}"].notna()
    df = df[keep].head(ocfg["board_depth"]).copy()
    df["overall_rank"] = np.arange(1, len(df) + 1)

    for f in FIELDS:
        if f not in df.columns:
            df[f] = np.nan

    news = load_news()
    auto = load_auto_news()
    df["note"], df["flag"] = zip(*[_news_for(r, news, auto) for _, r in df.iterrows()])

    int_fields = {"overall_rank", "tier", "bye_week", "n_sources",
                  "games_missed_l3y", "depth_rank"}
    players = []
    for _, row in df.iterrows():
        rec = {f: _clean(row[f]) for f in FIELDS}
        for f in int_fields:
            if isinstance(rec.get(f), float):
                rec[f] = int(rec[f])
        players.append(rec)

    manifest = read_manifest()
    sources = sorted(
        ({"key": k, "pulled_at": m.get("pulled_at"), "rows": m.get("rows"),
          "url": m.get("url", "")} for k, m in manifest.items()),
        key=lambda d: d["key"])

    repl = {str(n): {p: round(r, 1) for p, r in replacement_ranks(cfg, n).items()}
            for n in cfg["league"]["team_counts"]}

    meta = {
        "season": cfg["season"],
        "built": date.today().isoformat(),
        "teams": cfg["league"]["primary_team_count"],
        "team_counts": cfg["league"]["team_counts"],
        "roster": cfg["league"]["roster"],
        "positions": FANTASY_POS,
        "replacement": repl,
        "adp_teams": cfg["output"]["adp_teams"],
    }

    payload = json.dumps({"meta": meta, "players": players, "sources": sources},
                         separators=(",", ":"))

    html = _TEMPLATE.replace("__PAYLOAD__", payload)
    out_path = out_path or (OUTPUT / f"draft_board_{date.today().isoformat()}.html")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"  -> {out_path}  ({len(players)} players)")
    return str(out_path)


_TEMPLATE = r"""<title>2026 PPR Draft Board</title>
<style>
:root {
  color-scheme: light dark;
  --paper:#F6F8FA; --card:#FFFFFF; --ink:#10151C; --ink-2:#39434F;
  --slate:#5A6675; --line:#DDE3EA; --line-2:#EDF1F5;
  --accent:#B45309; --accent-2:#8A3E06;
  --good:#2E7D5B; --bad:#C0392B; --good-bg:#E4F1EA; --bad-bg:#FBE7E4;
  --t1:#B45309; --t2:#C2740A; --t3:#D08F0C; --t4:#DCA92B; --t5:#C9AE5A;
  --t6:#A9A98A; --t7:#8E9AA6; --t8:#7C8B99; --t9:#6E7D8C; --t10:#63707E;
  --shadow:0 1px 2px rgba(16,21,28,.06), 0 8px 24px rgba(16,21,28,.05);
  --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper:#0E1319; --card:#161D26; --ink:#E8EDF3; --ink-2:#B3BECC;
    --slate:#8593A3; --line:#26303C; --line-2:#1C242E;
    --accent:#FBBF24; --accent-2:#E0A215;
    --good:#4FBF8B; --bad:#F0705F; --good-bg:#163024; --bad-bg:#331C19;
    --t1:#FBBF24; --t2:#F0AC1C; --t3:#E09A18; --t4:#C98C22; --t5:#AE8437;
    --t6:#8F7C4F; --t7:#77808F; --t8:#6B7787; --t9:#616D7D; --t10:#576273;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.3);
  }
}
:root[data-theme="light"] {
  --paper:#F6F8FA; --card:#FFFFFF; --ink:#10151C; --ink-2:#39434F;
  --slate:#5A6675; --line:#DDE3EA; --line-2:#EDF1F5;
  --accent:#B45309; --accent-2:#8A3E06;
  --good:#2E7D5B; --bad:#C0392B; --good-bg:#E4F1EA; --bad-bg:#FBE7E4;
  --t1:#B45309; --t2:#C2740A; --t3:#D08F0C; --t4:#DCA92B; --t5:#C9AE5A;
  --t6:#A9A98A; --t7:#8E9AA6; --t8:#7C8B99; --t9:#6E7D8C; --t10:#63707E;
  --shadow:0 1px 2px rgba(16,21,28,.06), 0 8px 24px rgba(16,21,28,.05);
}
:root[data-theme="dark"] {
  --paper:#0E1319; --card:#161D26; --ink:#E8EDF3; --ink-2:#B3BECC;
  --slate:#8593A3; --line:#26303C; --line-2:#1C242E;
  --accent:#FBBF24; --accent-2:#E0A215;
  --good:#4FBF8B; --bad:#F0705F; --good-bg:#163024; --bad-bg:#331C19;
  --t1:#FBBF24; --t2:#F0AC1C; --t3:#E09A18; --t4:#C98C22; --t5:#AE8437;
  --t6:#8F7C4F; --t7:#77808F; --t8:#6B7787; --t9:#616D7D; --t10:#576273;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.3);
}

* { box-sizing:border-box; }
body {
  margin:0; background:var(--paper); color:var(--ink);
  font-family:var(--sans); font-size:14px; line-height:1.45;
  -webkit-font-smoothing:antialiased;
}
.wrap { max-width:1500px; margin:0 auto; padding:0 20px 72px; }

/* ---------- masthead ---------- */
header.mast { padding:32px 0 18px; border-bottom:2px solid var(--ink); margin-bottom:0; }
.mast-row { display:flex; align-items:flex-end; justify-content:space-between; gap:24px; flex-wrap:wrap; }
h1 {
  margin:0; font-size:clamp(28px,4.4vw,44px); font-weight:800;
  letter-spacing:-.03em; line-height:.98; text-wrap:balance;
}
h1 .yr { color:var(--accent); }
.dek { margin:8px 0 0; color:var(--slate); max-width:60ch; font-size:14px; }
.mast-meta { display:flex; gap:26px; flex-wrap:wrap; }
.mm { display:flex; flex-direction:column; gap:2px; }
.mm .k {
  font-size:10px; letter-spacing:.13em; text-transform:uppercase;
  color:var(--slate); font-weight:700;
}
.mm .v { font-family:var(--mono); font-size:15px; font-weight:600; font-variant-numeric:tabular-nums; }

/* ---------- scarcity strip ---------- */
.scarcity {
  position:sticky; top:0; z-index:30; background:var(--paper);
  border-bottom:1px solid var(--line); padding:10px 0 11px;
  display:flex; gap:8px; overflow-x:auto; scrollbar-width:thin;
}
.sc {
  flex:1 0 auto; min-width:132px; background:var(--card); border:1px solid var(--line);
  border-radius:7px; padding:8px 11px; display:flex; flex-direction:column; gap:3px;
  box-shadow:var(--shadow);
}
.sc.warn { border-color:var(--bad); background:var(--bad-bg); }
.sc-top { display:flex; align-items:baseline; justify-content:space-between; gap:8px; }
.sc-pos { font-weight:800; font-size:13px; letter-spacing:.04em; }
.sc-left { font-family:var(--mono); font-size:17px; font-weight:700; font-variant-numeric:tabular-nums; }
.sc-sub { font-size:11px; color:var(--slate); font-family:var(--mono); }
.sc.warn .sc-sub { color:var(--bad); font-weight:700; }

/* ---------- toolbar ---------- */
.bar {
  display:flex; gap:10px; align-items:center; flex-wrap:wrap;
  padding:14px 0; border-bottom:1px solid var(--line);
}
.seg { display:flex; gap:2px; background:var(--line-2); padding:3px; border-radius:8px; }
.seg button {
  font:inherit; font-size:12.5px; font-weight:700; letter-spacing:.02em;
  border:0; background:transparent; color:var(--ink-2);
  padding:6px 12px; border-radius:6px; cursor:pointer;
}
.seg button[aria-pressed="true"] { background:var(--card); color:var(--ink); box-shadow:var(--shadow); }
.seg button:focus-visible, .tgl:focus-visible, input:focus-visible, .row:focus-visible
  { outline:2px solid var(--accent); outline-offset:2px; }
input[type="search"] {
  font:inherit; padding:7px 12px; border-radius:8px; border:1px solid var(--line);
  background:var(--card); color:var(--ink); min-width:190px;
}
.tgl {
  font:inherit; font-size:12.5px; font-weight:700; padding:7px 13px; cursor:pointer;
  border-radius:8px; border:1px solid var(--line); background:var(--card); color:var(--ink-2);
}
.tgl[aria-pressed="true"] { background:var(--ink); color:var(--paper); border-color:var(--ink); }
.spacer { flex:1 1 auto; }
.count { font-family:var(--mono); font-size:12px; color:var(--slate); font-variant-numeric:tabular-nums; }

/* ---------- board table ---------- */
.scroller { overflow-x:auto; }
table { border-collapse:collapse; width:100%; min-width:1080px; }
thead th {
  position:sticky; top:57px; z-index:20; background:var(--paper);
  font-size:10px; letter-spacing:.11em; text-transform:uppercase; color:var(--slate);
  font-weight:800; text-align:right; padding:10px 8px 8px; border-bottom:1px solid var(--line);
  white-space:nowrap;
}
thead th.l { text-align:left; }
tbody td {
  padding:7px 8px; border-bottom:1px solid var(--line-2); text-align:right;
  font-family:var(--mono); font-variant-numeric:tabular-nums; font-size:12.5px;
  white-space:nowrap;
}
tbody td.l { text-align:left; font-family:var(--sans); }
.row { cursor:pointer; }
.row:hover td { background:var(--line-2); }
.row.drafted td { opacity:.4; }
.row.drafted .nm { text-decoration:line-through; }
.nm { font-weight:650; letter-spacing:-.005em; }
.rk { color:var(--slate); font-size:11.5px; }
.rail { padding:0 !important; width:5px; border-bottom:0 !important; }
.rail i { display:block; width:5px; height:100%; min-height:30px; background:var(--tc); }
.pos {
  display:inline-block; min-width:34px; text-align:center; font-family:var(--mono);
  font-size:10.5px; font-weight:800; letter-spacing:.06em; padding:2px 5px;
  border-radius:4px; border:1px solid var(--line); color:var(--ink-2); background:var(--card);
}
.tierbadge {
  font-family:var(--mono); font-size:11px; font-weight:800; color:var(--tc);
  letter-spacing:.02em;
}
.eml { font-size:10px; letter-spacing:.09em; text-transform:uppercase; color:var(--slate); font-weight:700; }
.chip {
  display:inline-block; min-width:52px; text-align:center; padding:2px 6px; border-radius:5px;
  font-size:11.5px; font-weight:700; font-family:var(--mono);
}
.chip.v { background:var(--good-bg); color:var(--good); }
.chip.r { background:var(--bad-bg); color:var(--bad); }
.chip.n { color:var(--slate); }
.muted { color:var(--slate); }
.note { max-width:230px; white-space:normal; font-size:11.5px; line-height:1.3;
  color:var(--ink-2); }
.note-out { color:var(--bad); font-weight:600; }
.note-down { color:#b26a00; font-weight:600; }
.note-watch { color:#8a6d00; }
.note-up { color:var(--good); font-weight:600; }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%;
  margin-right:6px; vertical-align:middle; }
.dot-out { background:var(--bad); }
.dot-down { background:#e08a00; }
.dot-watch { background:#e0b400; }
.dot-up { background:var(--good); }

/* the tier cliff: height and label encode the real VBD drop */
tr.cliff td { padding:0; border:0; }
.cliffbar {
  display:flex; align-items:center; gap:10px; padding:3px 8px 3px 13px;
  background:linear-gradient(90deg,var(--tc) 0%,transparent 62%);
  background-color:var(--line-2);
}
.cliffbar .lab {
  font-size:10px; letter-spacing:.13em; text-transform:uppercase; font-weight:800; color:var(--ink);
}
.cliffbar .drop { font-family:var(--mono); font-size:11px; font-weight:700; color:var(--ink-2); }

/* ---------- cheat sheet ---------- */
.cheat { display:grid; grid-template-columns:repeat(auto-fit,minmax(215px,1fr)); gap:16px; padding-top:18px; }
.col h3 {
  margin:0 0 8px; font-size:12px; letter-spacing:.13em; text-transform:uppercase;
  font-weight:800; padding-bottom:6px; border-bottom:2px solid var(--ink);
}
.cl { display:flex; align-items:center; gap:7px; padding:3px 0 3px 8px; border-left:4px solid var(--tc); }
.cl.drafted { opacity:.35; text-decoration:line-through; }
.cl .n { flex:1 1 auto; font-size:12.5px; font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.cl .t, .cl .a {
  font-family:var(--mono); font-size:10.5px; color:var(--slate); font-variant-numeric:tabular-nums;
}
.cbreak { height:9px; }

/* ---------- sources ---------- */
.src { padding-top:18px; }
.src table { min-width:760px; }
.src td, .src th { text-align:left; }
.src td.u { font-size:11px; color:var(--slate); max-width:620px; overflow:hidden; text-overflow:ellipsis; }
.note { color:var(--slate); font-size:12.5px; max-width:72ch; margin:14px 0 0; }
footer { margin-top:34px; padding-top:16px; border-top:1px solid var(--line); color:var(--slate); font-size:12px; }
[hidden] { display:none !important; }
@media (prefers-reduced-motion:reduce) { * { transition:none !important; animation:none !important; } }
</style>

<div class="wrap">
<header class="mast">
  <div class="mast-row">
    <div>
      <h1><span class="yr" id="hSeason"></span> PPR Draft Board</h1>
      <p class="dek">Ranked by value over replacement, not raw points. Tiers are
      cut where the real cliffs are. Click any player to mark them drafted —
      the scarcity strip recalculates.</p>
    </div>
    <div class="mast-meta" id="mastMeta"></div>
  </div>
</header>

<div class="scarcity" id="scarcity"></div>

<div class="bar">
  <div class="seg" id="viewSeg">
    <button data-view="board" aria-pressed="true">Board</button>
    <button data-view="cheat" aria-pressed="false">Cheat sheet</button>
    <button data-view="sources" aria-pressed="false">Sources</button>
  </div>
  <div class="seg" id="posSeg"></div>
  <input type="search" id="q" placeholder="Search player or team" aria-label="Search players">
  <button class="tgl" id="hideDrafted" aria-pressed="false">Hide drafted</button>
  <button class="tgl" id="newsOnly" aria-pressed="false">News only</button>
  <button class="tgl" id="reset">Reset</button>
  <span class="spacer"></span>
  <span class="count" id="count"></span>
</div>

<section id="vBoard"><div class="scroller"><table>
  <thead><tr>
    <th class="l" style="width:5px"></th>
    <th class="l">Rk</th><th class="l">Player</th><th class="l">Pos</th><th class="l">Tm</th>
    <th class="l">Tier</th><th class="l">E/M/L</th>
    <th>Proj</th><th>VBD</th><th>8tm</th><th>10tm</th>
    <th>Floor</th><th>Ceil</th><th>ADP</th><th>Value</th>
    <th>Bye</th><th>SOS</th><th>Age</th><th>Src</th><th class="l">News / Notes</th>
  </tr></thead>
  <tbody id="tb"></tbody>
</table></div></section>

<section id="vCheat" hidden><div class="cheat" id="cheat"></div></section>

<section id="vSources" hidden><div class="src">
  <div class="scroller"><table>
    <thead><tr><th class="l">Dataset</th><th class="l">Pulled at (UTC)</th><th class="l">Rows</th><th class="l">Source</th></tr></thead>
    <tbody id="srcTb"></tbody>
  </table></div>
  <p class="note" id="replNote"></p>
</div></section>

<footer id="foot"></footer>
</div>

<script>
const DATA = __PAYLOAD__;
const P = DATA.players, M = DATA.meta;
const drafted = new Set();
let view = "board", pos = "ALL", q = "", hideDrafted = false, newsOnly = false;

const FLAG_LABEL = {out:"injury / downgrade", down:"downgrade",
                    watch:"volatile - watch", up:"upgrade / rising"};
function flagDot(flag) {
  if (!flag) return "";
  return `<span class="dot dot-${flag}" title="${FLAG_LABEL[flag]||flag}"></span>`;
}

const $ = s => document.querySelector(s);
const tierVar = t => `var(--t${Math.min(Math.max(t|0,1),10)})`;
const num = (v,d=1) => v===null||v===undefined ? "" : v.toFixed(d);

/* masthead */
$("#hSeason").textContent = M.season;
const r = M.roster;
$("#mastMeta").innerHTML = [
  ["League", `${M.teams} team`],
  ["Starters", `${r.QB}QB ${r.RB}RB ${r.WR}WR ${r.TE}TE ${r.FLEX}FLX`],
  ["Players", P.length],
  ["Built", M.built],
].map(([k,v]) => `<div class="mm"><span class="k">${k}</span><span class="v">${v}</span></div>`).join("");
$("#foot").textContent =
  `Replacement level at ${M.teams} teams: ` +
  M.positions.map(p => `${p}${Math.round(M.replacement[M.teams][p])}`).join(" · ") +
  `. ADP is ${M.adp_teams}-team PPR. Value is in rounds: positive means the market is letting them fall.`;
$("#replNote").textContent =
  "Every number on this board traces back to one of these pulls. " +
  "nflverse injuries for the current season are absent until the first in-season " +
  "report is published, so durability comes from games missed over the last three years.";

/* position filter */
$("#posSeg").innerHTML = ["ALL", ...M.positions]
  .map(p => `<button data-pos="${p}" aria-pressed="${p==="ALL"}">${p==="ALL"?"All":p}</button>`).join("");

/* ---------- rendering ---------- */
function visible() {
  const needle = q.trim().toLowerCase();
  return P.filter(p => {
    if (pos !== "ALL" && p.position !== pos) return false;
    if (hideDrafted && drafted.has(p.overall_rank)) return false;
    if (newsOnly && !p.note) return false;
    if (needle) {
      const hay = (p.player_name + " " + (p.team||"") + " " + p.position).toLowerCase();
      if (!hay.includes(needle)) return false;
    }
    return true;
  });
}

function valueChip(v) {
  if (v === null || v === undefined) return `<span class="chip n">—</span>`;
  const cls = v >= 0.75 ? "v" : v <= -0.75 ? "r" : "n";
  const sign = v > 0 ? "+" : "";
  return `<span class="chip ${cls}">${sign}${v.toFixed(1)}</span>`;
}

function renderBoard() {
  const rows = visible();
  const out = [];
  let prevTier = null, prevPos = null, prevVbd = null;
  for (const p of rows) {
    const tc = tierVar(p.tier);
    // A tier change inside one position is a genuine cliff: show the drop.
    if (pos !== "ALL" && prevTier !== null && p.tier !== prevTier && p.position === prevPos) {
      const drop = prevVbd !== null && p.vbd_score !== null ? prevVbd - p.vbd_score : null;
      out.push(`<tr class="cliff"><td colspan="19"><div class="cliffbar" style="--tc:${tc}">
        <span class="lab">${p.position} tier ${p.tier}</span>
        <span class="drop">${drop!==null?"▼ "+drop.toFixed(1)+" VBD cliff":""}</span>
      </div></td></tr>`);
    }
    prevTier = p.tier; prevPos = p.position; prevVbd = p.vbd_score;
    const isD = drafted.has(p.overall_rank);
    out.push(`<tr class="row${isD?" drafted":""}" data-rk="${p.overall_rank}" tabindex="0" style="--tc:${tc}">
      <td class="rail"><i></i></td>
      <td class="l rk">${p.overall_rank}</td>
      <td class="l nm">${flagDot(p.flag)}${p.player_name}</td>
      <td class="l"><span class="pos">${p.position}</span></td>
      <td class="l muted">${p.team||""}</td>
      <td class="l"><span class="tierbadge">${p.tier_label||""}</span></td>
      <td class="l"><span class="eml">${p.sub_tier_label||""}</span></td>
      <td>${num(p.final_projection)}</td>
      <td><b>${num(p.vbd_score)}</b></td>
      <td class="muted">${num(p.vbd_8)}</td>
      <td class="muted">${num(p.vbd_10)}</td>
      <td class="muted">${num(p.projection_low)}</td>
      <td class="muted">${num(p.projection_high)}</td>
      <td>${num(p.adp_10)}</td>
      <td>${valueChip(p.value_rounds)}</td>
      <td class="muted">${p.bye_week??""}</td>
      <td class="muted">${num(p.sos_z,2)}</td>
      <td class="muted">${num(p.age)}</td>
      <td class="muted">${p.n_sources??""}</td>
      <td class="l note ${p.flag?("note-"+p.flag):""}">${p.note||""}</td>
    </tr>`);
  }
  $("#tb").innerHTML = out.join("");
  $("#count").textContent = `${rows.length} shown · ${drafted.size} drafted`;
}

function renderCheat() {
  const needle = q.trim().toLowerCase();
  const cols = M.positions.filter(p => pos === "ALL" || p === pos);
  $("#cheat").innerHTML = cols.map(pp => {
    const list = P.filter(p => p.position === pp)
      .filter(p => !(hideDrafted && drafted.has(p.overall_rank)))
      .filter(p => !needle || (p.player_name+" "+(p.team||"")).toLowerCase().includes(needle));
    let prev = null;
    const items = list.map(p => {
      const br = (prev !== null && p.tier !== prev) ? `<div class="cbreak"></div>` : "";
      prev = p.tier;
      const isD = drafted.has(p.overall_rank);
      return `${br}<div class="cl${isD?" drafted":""}" data-rk="${p.overall_rank}"
          style="--tc:${tierVar(p.tier)}">
        <span class="t">${p.tier_label||""}</span>
        <span class="n">${p.player_name}</span>
        <span class="a">${p.adp_10!==null?p.adp_10.toFixed(0):"—"}</span>
      </div>`;
    }).join("");
    return `<div class="col"><h3>${pp}</h3>${items}</div>`;
  }).join("");
}

function renderScarcity() {
  $("#scarcity").innerHTML = M.positions.map(pp => {
    const live = P.filter(p => p.position === pp && !drafted.has(p.overall_rank));
    if (!live.length) {
      return `<div class="sc warn"><div class="sc-top"><span class="sc-pos">${pp}</span>
        <span class="sc-left">0</span></div><div class="sc-sub">board empty</div></div>`;
    }
    const best = Math.min(...live.map(p => p.tier));
    const inTier = live.filter(p => p.tier === best).length;
    const warn = inTier <= 2;
    return `<div class="sc${warn?" warn":""}">
      <div class="sc-top"><span class="sc-pos">${pp}</span><span class="sc-left">${live.length}</span></div>
      <div class="sc-sub">${warn?"⚠ ":""}${pp}${best} · ${inTier} left</div>
    </div>`;
  }).join("");
}

function renderSources() {
  $("#srcTb").innerHTML = DATA.sources.map(s => `<tr>
    <td class="l">${s.key}</td>
    <td class="l">${s.pulled_at||""}</td>
    <td class="l">${s.rows!=null?s.rows.toLocaleString():""}</td>
    <td class="l u">${(s.url||"").split("; ")[0]}</td></tr>`).join("");
}

function render() {
  renderScarcity();
  $("#vBoard").hidden = view !== "board";
  $("#vCheat").hidden = view !== "cheat";
  $("#vSources").hidden = view !== "sources";
  if (view === "board") renderBoard();
  else if (view === "cheat") renderCheat();
  else renderSources();
}

/* ---------- interaction ---------- */
function toggle(rk) {
  rk = Number(rk);
  drafted.has(rk) ? drafted.delete(rk) : drafted.add(rk);
  render();
}
document.addEventListener("click", e => {
  const row = e.target.closest(".row, .cl");
  if (row) { toggle(row.dataset.rk); return; }
  const vb = e.target.closest("#viewSeg button");
  if (vb) {
    view = vb.dataset.view;
    document.querySelectorAll("#viewSeg button")
      .forEach(b => b.setAttribute("aria-pressed", String(b === vb)));
    render(); return;
  }
  const pb = e.target.closest("#posSeg button");
  if (pb) {
    pos = pb.dataset.pos;
    document.querySelectorAll("#posSeg button")
      .forEach(b => b.setAttribute("aria-pressed", String(b === pb)));
    render();
  }
});
document.addEventListener("keydown", e => {
  if (e.key === "Enter" || e.key === " ") {
    const row = e.target.closest?.(".row");
    if (row) { e.preventDefault(); toggle(row.dataset.rk); }
  }
});
$("#q").addEventListener("input", e => { q = e.target.value; render(); });
$("#hideDrafted").addEventListener("click", e => {
  hideDrafted = !hideDrafted;
  e.currentTarget.setAttribute("aria-pressed", String(hideDrafted));
  render();
});
$("#newsOnly").addEventListener("click", e => {
  newsOnly = !newsOnly;
  e.currentTarget.setAttribute("aria-pressed", String(newsOnly));
  render();
});
$("#reset").addEventListener("click", () => {
  drafted.clear(); q = ""; $("#q").value = "";
  hideDrafted = false; $("#hideDrafted").setAttribute("aria-pressed","false");
  newsOnly = false; $("#newsOnly").setAttribute("aria-pressed","false");
  render();
});

render();
</script>
"""


def main(argv=None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Render the draft board as HTML")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    print("=== Output: html ===")
    build(cfg, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
