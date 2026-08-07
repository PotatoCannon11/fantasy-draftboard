# Fantasy Football Draft System — Build Spec

## Project Goal
Build a full research-to-draft-board pipeline that produces a tiered, print/glance-ready
draft board spreadsheet, refreshed with current data in the days before a live draft.
Goal is a repeatable system, not a one-off spreadsheet — should be re-runnable every year
with a single data refresh.

## League Settings (confirmed)
- Format: **PPR** (point per reception)
- Type: Standard redraft (not dynasty/keeper/superflex)
- Teams: **8-10 team league**
- Roster: 1 QB, 2 RB, 2 WR, 1 FLEX (RB/WR/TE), 1 TE, 1 K, 1 DEF
- Draft timing: always ~1-2 weeks before NFL regular season kickoff (post roster-cuts,
  post most camp battles resolving). Exact date TBD each year — system should support
  "refresh as of date X" as a parameter.

## Design Principles
1. **Don't over-engineer the model.** Fantasy football has a small annual sample size
   (~17 games/season). A single from-scratch deep learning model will likely overfit.
   Favor an ensemble/ranking-blend approach over one heavy custom model.
2. **Separate layers cleanly:** Data ingestion → Feature engineering → Projection/value
   model → Tiering → Spreadsheet output. Each layer should be independently re-runnable.
3. **Position-relative value, not raw points.** Use VBD/VORP (Value Based Drafting),
   not raw projected points, as the core ranking metric.
4. **Tiers from score gaps, not fixed round cutoffs.** Detect real cliffs in value,
   not arbitrary "every 12 players."
5. **Weather is explicitly excluded from the draft-day model** — forecasts aren't
   usable 2 weeks out. This is a week-to-week in-season tool only, out of scope here.
6. **Log data source + pull timestamp for every dataset used**, so it's clear at
   draft time how fresh each input is.

---

## Layer 1: Data Ingestion

Primary free data backbone: **nflverse**
- Python: `nfl-data-py`
- R: `nflreadr` (equivalent, pick one language — Python preferred for this project)

Pull functions needed (Python `nfl_data_py` equivalents):
- `import_rosters()` — current rosters
- `import_depth_charts()` — depth chart position/ranking per team
- `import_injuries()` — current injury report + historical injury/games-missed data
- `import_snap_counts()` — snap % by player/week (trend signal)
- `import_ngs_data()` — Next Gen Stats: separation, aDOT, YAC over expectation,
  pressure rate, completion probability
- `import_seasonal_data()` / `import_weekly_data()` — box score stats, historical
- `import_schedules()` — for SOS calculation and bye weeks
- `import_ids()` — player ID mapping across sources (needed to join datasets cleanly)

Secondary source: **ffopportunity** (nflverse sister project)
- Provides pre-built Expected Fantasy Points via an XGBoost model trained on
  play-by-play data — use this as an "opportunity quality" feature rather than
  rebuilding an expected-points model from scratch.
- R package `ffopportunity`; if Python-only, either call via subprocess/rpy2 or
  pull their precomputed data releases directly (check their GitHub releases for
  CSV/parquet exports).

Manual/scraped sources (no clean API — build a simple scraper or manual CSV update):
- ADP: FantasyPros ADP pages (filter for PPR, appropriate league size)
- Scheme/coaching tags: manual qualitative tagging per team (see Layer 2)
- O-line grades: PFF free content or Football Outsiders adjusted line yards as fallback

**Deliverable for this layer:** a `data/raw/` directory of pulled datasets, each with
a `pulled_at` timestamp, and a `refresh_data.py` script that re-pulls everything on
demand (parameterized by as-of date).

---

## Layer 2: Feature Engineering

Build a per-player, per-season (or per-week, for trend features) feature table.

**Volume/opportunity features**
- `target_share`, `air_yards_share`, `wopr` (weighted opportunity rating =
  function of target share + air yards share)
- `carry_share`, `redzone_touch_share`, `goalline_share`
- `snap_pct_trend` — snap % over last 4 games of prior season (weighted more than
  season average)
- `exp_fantasy_points` (from ffopportunity) and `exp_pts_delta` = actual − expected
  (regression candidate signal — big positive delta = due for negative regression,
  big negative delta = breakout/positive regression candidate)

**Efficiency features**
- `yprr` (yards per route run), `yac_oe` (YAC over expectation), `catch_rate_oe`,
  `separation` (from NGS)
- `rush_efficiency_oe` — rushing yards over expectation, adjusted for O-line grade
  (don't credit RB for O-line, don't blame RB for O-line)

**Context/scheme features (mostly manually tagged, categorical)**
- `team_proe` — pass rate over expected (tags team as pass-funnel vs run-funnel)
- `target_concentration_tag` — does this offense concentrate targets on one WR or
  spread them (derive from historical target share distribution across the team's
  pass catchers)
- `oline_grade`
- `qb_tier` — categorical multiplier applied to that team's pass catchers

**Risk/durability features**
- `games_missed_l3y` — games missed over last 3 seasons
- `age_curve_position` — position-specific age discount (RBs decline earlier/steeper
  than WRs — apply real, non-trivial discount past ~age 27 for RBs specifically)
- `contract_year_flag` (small weight, soft signal)

**Schedule feature (applied later, NOT baked into base projection)**
- `sos_adjusted` — defense-adjusted strength of schedule by position, weighted
  toward weeks 1-6 (near-term) more than weeks 15-17 (distant/unknowable)

**Deliverable for this layer:** `features.py` that joins Layer 1 raw data into one
clean `player_features` table, one row per player.

---

## Layer 3: Projection Model

Approach, in priority order (build Tier 1 first, it's the highest ROI):

**Tier 1 — Wisdom-of-crowds blend (build this first)**
- Pull 3-5 public projection sets (FantasyPros consensus, and 2-3 others if
  accessible) for the current season, PPR-specific
- Z-score normalize each set within position
- Average into a single blended baseline projection per player

**Tier 2 — Context adjustment layer (this is where the research edge comes from)**
- Apply multipliers to the Tier 1 baseline using Layer 2 features:
  `final_projection = blended_baseline * scheme_multiplier * injury_discount *
  target_competition_adjustment * exp_pts_regression_signal`
- Document each multiplier's formula clearly and keep them tunable (config file,
  not hardcoded) so the person can adjust weights year to year based on what
  they learn.

**Tier 3 — Optional custom regression model**
- If desired: ridge or elastic net regression (NOT a deep net — data size doesn't
  support it) trained on 3-5 years of nflverse historical data, predicting season
  fantasy points from Layer 2 features
- Use as one more input into the Tier 1 ensemble blend, not a standalone replacement
- Random forest is a reasonable alternative to try alongside ridge/elastic net;
  compare via cross-validated RMSE against the public projection sets before trusting it

**Deliverable for this layer:** `project_players.py` producing a `final_projection`
and `projection_stddev` (see Layer 4) per player.

---

## Layer 4: Uncertainty / Variance Modeling

For each player, compute a variance/range estimate driven by:
- Role clarity (locked-in starter = tight variance; committee/rookie/new-situation
  = wide variance)
- Injury history
- Target/touch competition on the roster

Optional (nice-to-have, not required for v1): Monte Carlo simulation — simulate
each player's season N times sampling from a game-level distribution informed by
the variance estimate, to get a full outcome distribution rather than a single
point estimate. Useful for defensible floor/ceiling labeling.

**Deliverable:** each player gets `projection_low`, `projection_mid`, `projection_high`
or equivalent variance measure.

---

## Layer 5: Value Conversion (VBD) and Tiering

**VBD/VORP calculation:**
- For each position, determine the "replacement level" player = the last
  startable player at that position given league settings (8-10 teams, roster
  slots above, accounting for FLEX-eligible RB/WR/TE competing for that slot)
- `vbd_score = player_projection - replacement_level_projection` (per position)

**Tiering algorithm:**
- Rank players within position by VBD score
- Detect tier breaks via score-gap analysis: flag a new tier when the gap to the
  next player exceeds some threshold relative to local score variance (e.g. gap >
  0.5 * rolling stddev of scores in that neighborhood) — OR use k-means clustering
  on VBD score per position as an alternative/cross-check
- Within each tier, sort into **Early / Mid / Late** sub-labels using the Layer 4
  variance data: Early = highest floor within the tier, Late = highest
  variance/risk within the tier

**FLEX overlay:** since this league has only 1 FLEX (RB/WR/TE), also produce a
combined RB/WR/TE value scale (not just within-position) for players near the
flex-relevant value range, so RB vs WR flex decisions are visible.

**Deliverable:** `tiering.py` producing final `tier`, `sub_tier_label`, and
`vbd_score` columns.

---

## Output: Spreadsheet

Build a spreadsheet (xlsx) with these tabs:

1. **Master Board** — player, position, team, blended projection, VBD score, tier,
   sub-tier label (Early/Mid/Late), ADP, value-vs-ADP delta (flag values and reaches),
   injury flag, bye week, projection_low/mid/high
2. **Position tabs** (QB/RB/WR/TE/K/DEF) — same data filtered per position
3. **Live Draft Tracker** — checkbox/status column to mark players drafted; should
   auto-recalculate remaining positional scarcity and flag positional runs
   (e.g., detect when N of the last M picks were the same position and warn that a
   tier is about to dry up)
4. **Cheat Sheet / Print View** — condensed, tier-color-coded, designed for fast
   glancing under draft-clock pressure — this is the primary in-draft artifact

Formatting: color-code by tier, bold tier breaks, conditional formatting on
value-vs-ADP delta (green = value, red = reach).

---

## Suggested Tech Stack
- Python 3.x
- pandas, numpy for data wrangling
- scikit-learn (ridge/elastic net/random forest) if building Tier 3 model
- `nfl_data_py` for data ingestion
- `openpyxl` or `xlsxwriter` for the final spreadsheet output with formatting
- Simple CLI or notebook-driven workflow — no need for a web app/UI, this is a
  personal tool run a handful of times per year

## Suggested Repo Structure
```
fantasy-draft-system/
├── data/
│   ├── raw/              # pulled datasets, timestamped
│   └── processed/        # feature tables
├── src/
│   ├── ingest.py         # Layer 1
│   ├── features.py       # Layer 2
│   ├── project.py        # Layer 3
│   ├── variance.py       # Layer 4
│   ├── tiering.py        # Layer 5
│   └── build_spreadsheet.py
├── config/
│   └── weights.yaml       # tunable multipliers, VBD replacement-level settings, tier-gap thresholds
├── output/
│   └── draft_board_<date>.xlsx
└── README.md
```

## Refresh Workflow (run relative to draft date)
- **10-14 days out:** pull nflverse core datasets, build base model/projections
  (this data is stable, doesn't need re-touching)
- **5-7 days out:** refresh depth charts/rosters (post roster-cuts), pull current ADP
- **48 hrs out:** final injury report pull, final depth chart check, lock ADP snapshot
- **Night before:** last-minute news scan only, flag specific affected players —
  should not require re-running the full pipeline, just a manual flag/override
  column in the spreadsheet

## Explicitly Out of Scope for v1
- Weather (not usable this far out from games; revisit as an in-season tool)
- In-season lineup/start-sit tooling
- Auction-draft dollar values (build only if league switches formats)
- Web UI — CLI/notebook + spreadsheet output is sufficient
