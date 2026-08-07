# Fantasy Football Draft System

Research-to-draft-board pipeline. Pulls current NFL data and market
projections, blends them, adjusts for context, converts to position-relative
value, detects tier cliffs, and writes a print-ready xlsx draft board.

Built to the spec in `fantasy_football_draft_system_spec.md`. Re-runnable every
year with a single data refresh.

## Quick start

```bash
.venv/bin/python -m fantasydraft.run_pipeline      # full refresh + build (~3 min)
.venv/bin/python -m fantasydraft.validate          # 38 data-integrity checks
.venv/bin/python -m fantasydraft.verify_xlsx       # 38 LIVE formula checks via LibreOffice
```

Output lands in `output/draft_board_<date>.xlsx` (plus a browser-viewable
`.html` of the same board).

`fantasydraft/verify_xlsx.py` matters more than it sounds. xlsxwriter stores a placeholder
`0` for every formula it writes, so reading the file back with openpyxl proves
nothing about whether the formulas work. This drives a real LibreOffice engine
over the workbook with recalculate-on-load forced on, and asserts against the
computed values. It is what caught the `MINIFS` bug below. Set `FDS_SOFFICE` if
LibreOffice is not on your PATH.

The virtualenv is built on Python 3.11 (`requirements.txt` is pinned loosely).
To recreate it:

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Re-running individual layers

Every layer reads the parquet the previous one wrote, so you never have to
re-pull nflverse just to retune a weight.

```bash
python fantasydraft/run_pipeline.py --refresh market   # just ADP + projections
python fantasydraft/run_pipeline.py --stage project    # re-blend onwards, no re-download
python fantasydraft/run_pipeline.py --stage spreadsheet
python src/features.py                    # any layer runs standalone too
```

## The layers

| Layer | File | What it does |
|---|---|---|
| 1. Ingest | `fantasydraft/ingest.py` | Pulls every dataset into `data/raw/`, logging source URL + timestamp per file into `_manifest.json` |
| 2. Features | `fantasydraft/features.py` | Joins raw data into `player_features` + `team_features`; regenerates `config/team_context.yaml` |
| 3. Projections | `fantasydraft/project.py` | Tier 1 crowd blend, then Tier 2 context multipliers |
| 4. Variance | `fantasydraft/variance.py` | Per-player sigma → floor / mid / ceiling |
| 5. VBD + tiers | `fantasydraft/tiering.py`, `fantasydraft/board.py` | Replacement level, VBD, gap-based tiers, ADP join |
| Output | `fantasydraft/build_spreadsheet.py` | The 10-tab xlsx |

Supporting: `fantasydraft/common.py` (paths, name/team normalization), `fantasydraft/idmap.py`
(cross-source player identity).

## Data sources

All free, no API keys.

| What | Source | Notes |
|---|---|---|
| Core NFL data | [nflverse-data](https://github.com/nflverse/nflverse-data) releases | rosters, depth charts, snap counts, NGS, play-by-play, schedules, seasonal + weekly stats |
| Expected fantasy points | [ffopportunity](https://github.com/ffverse/ffopportunity) | pre-built XGBoost expected-points model, used as a feature |
| Player ID crosswalk | [DynastyProcess](https://github.com/dynastyprocess/data) | maps sleeper / espn / gsis / fantasypros ids |
| ADP | FantasyFootballCalculator API | PPR, pulled for both 8- and 10-team |
| Projections | Sleeper, ESPN, FantasySharks | three independent sets, blended |

**Note on `nfl_data_py`:** deliberately not used. It is unmaintained and pins
`pandas<2` / `numpy<2`, which conflicts with everything else here. `ingest.py`
reads the same nflverse release assets directly.

**Note on FantasyPros:** their projection pages are now JS-gated to 10 rows per
position, so they cannot be scraped for a full projection set. FantasySharks
fills that slot. If you get FantasyPros access, add it to `SOURCES` in
`fantasydraft/project.py` and give it a weight in `config/weights.yaml`.

## Tuning

Everything judgemental lives in `config/weights.yaml` — nothing is hardcoded.

- `league.team_counts` — VBD is computed for every size listed; the board shows
  `VBD 8tm` and `VBD 10tm` side by side. `primary_team_count` drives sort order
  and tiering.
- `projections.sources` — relative source weights, renormalized over whichever
  sources actually have a given player.
- `context.*` — each Tier 2 multiplier, individually switchable. The compounded
  product is clamped to `max_total_swing` (default ±20%) so no single player can
  run away from the consensus.
- `variance.*` — base coefficient of variation per position and what widens it.
- `vbd.streaming_allowance` / `flex_split` — how replacement level is derived.
- `tiering.gap_threshold` — raise it for fewer, larger tiers.

`config/team_context.yaml` is the hand-editable team overlay. The `derived`
block is regenerated on every `features.py` run; the `override` block is never
touched. Set a value under `override` to overrule the derived number — that is
where you encode offseason news the 2025 data cannot know about:

```yaml
teams:
  CLE:
    derived: {qb_tier: 5, oline_z: -0.31, proe_z: -0.66, ...}
    override:
      qb_tier: 3          # this wins
      note: "rookie QB looked good in camp"
```

## How the model works

**Tier 1 — crowd blend.** Each source is z-scored *within position*, the
z-scores are averaged with configured weights, then mapped back onto the pooled
points scale. Z-scoring first stops a source that is simply scaled differently
(e.g. one assuming 16 games) from dragging the blend.

**Tier 2 — context.** Multipliers built from Layer 2 features, each centred on
1.0 and compounded:

| Multiplier | Applies to | Signal |
|---|---|---|
| `m_proe` | WR/TE, inverse for RB | team neutral-situation pass rate |
| `m_exp_regression` | all | last season's actual − expected fantasy points; big overperformers get discounted |
| `m_target_competition` | WR/TE | how concentrated the team's target tree is, helping the alpha and hurting the rest |
| `m_depth_chart` | skill positions | current depth-chart rank |
| `m_injury` | all | games missed over the last 3 seasons |
| `m_age` | all | position-specific age curve (RB discount starts at 26 and is steep) |
| `m_qb_tier` | WR/TE | quality of the team's passer |
| `m_oline` | RB | team rushing-yards-over-expected proxy |

**Tier 3** (custom ridge/elastic-net regression) is intentionally not built.
The blend plus context layer is the high-ROI part; a from-scratch model on ~17
games a season mostly overfits. If you want it later, train on
`data/processed/player_features.parquet` against historical
`fantasy_points_ppr`, cross-validate RMSE against the public sources first, and
only then add it as a fourth entry in `SOURCES`.

**Variance.** Base CV per position, widened by unclear role, rookie status, a
new team, injury history, and cross-source disagreement. Drives floor/ceiling
and the Early/Mid/Late label.

**VBD.** Replacement level is derived from your actual settings — starters, plus
the position's expected share of the single FLEX slot, plus a streaming
allowance — not a rule of thumb. At 10 teams that lands on QB15 / RB40 / WR40 /
TE15, and the board interpolates between the two players bracketing a
fractional rank.

**Tiering.** A new tier opens where the drop to the next player exceeds
`gap_threshold ×` the local rolling stddev of *scores*. K-means is available as
a cross-check (`tiering.method: kmeans`); the gap method also records
`tier_kmeans` so you can see where the two disagree.

## What the historical data says (and what changed because of it)

Everything below was measured on nflverse 2014–2025, not asserted. The scripts
that produced these numbers are throwaway analyses; the conclusions are baked
into `config/weights.yaml` with the reasoning in comments.

**Replacement level was the biggest error.** The theoretical starters + FLEX +
streaming formula put WR replacement at WR40. Counting what real drafters
actually do in the FFC ADP sample (2,873 drafts) shows **63 WRs** come off the
board in a 10-team league. Baselining at WR40 uses a far better player as
"replacement" and systematically understates every WR. Fixing it rebalanced the
top 50 from RB-dominated to 24 RB / 23 WR and removed a bogus TE premium.
`vbd.method: adp_demand` now derives this from ADP each year.

**The context layer earns its keep.** On held-out seasons (2022–24), age +
TD-regression multipliers beat naive persistence by **4.9% RMSE**; each helps
~2.5% alone and they are complementary. TD regression is monotonic — the
highest-TD-rate quintile declines **−16.7%** the next year vs **−4.6%** for the
lowest.

**Variance was understated by about a third.** Measured forecast error is
QB .25 / RB .42 / WR .40 / TE .42; the config said .18/.30/.28/.30. Floor and
ceiling bands are now honest, which makes them wider.

**Age curves were wrong, and the naive way of measuring them is a trap.** Mean
PPG by age says RBs peak at 22 — pure survivorship, since only the good ones are
still playing at 31. Measured *within player*, RB decline starts ~24 and is
steep; QBs hold value to ~30.

**Efficiency stats are weak predictors.** Opportunities correlate +0.40 with
next-season PPG; yards per opportunity only +0.18 and TD rate +0.12. The o-line
multiplier was halved accordingly.

**Shrinkage toward the positional mean** is worth 3.5% on raw last-year PPG at
k=0.30 — but that does *not* transfer, because public projections already
regress internally. Applied at k=0.06 (0.18 for single-source players).

Not implemented: a proper BEER/man-games baseline. It needs expected games by
*preseason projected* rank, and measuring that from realized rank is circular
(low ranks are low *because* of missed games). Noted rather than faked.

## The spreadsheet

- **Master Board** — everything, sorted by VBD, autofiltered. Conditional
  formatting: green = value vs ADP, red = reach.
- **QB / RB / WR / TE / K / DEF** — same columns, filtered per position.
- **Live Draft Tracker** — put any mark in the `Drafted?` column and the row
  strikes through. The scarcity panel (players left, best tier still available,
  how many are in it) and the "TIER DRYING UP" warning are worksheet formulas,
  so they recalculate live with no Python running.
- **Cheat Sheet** — all six positions side by side, tier-coloured, fits one page
  wide. This is the in-draft artifact.
- **Compare** — pick up to four players from dropdowns; every metric lines up
  side by side with the best value in each row highlighted. Uses INDEX/MATCH
  against the Master Board, so it stays live.
- **Pick Assistant** — two independent halves, side by side:
  - *Left, the plan*: Monte Carlo. Choose a pick number and it ranks players by
    **VONA**, showing the chance each is on the board now, the chance he
    survives to your next pick, and a TAKE NOW / probably gone / can wait /
    long shot verdict. Below it, expected best-available VBD per position now
    vs next pick — the biggest **Drop** is where the pick should go.
  - *Right, LIVE*: recomputed entirely from the `Drafted?` marks on the Live
    Draft Tracker. Best available at each position, live VONA, and one headline
    call (`TAKE: <player>`). Nothing here is precomputed — mark a player off and
    every cell moves. With an empty board it agrees with the plan; the moment
    your league goes off-script, this is the half that stays correct.

  The live half has an **Override** cell for the pick count, because you will
  fall behind on marking players mid-draft. Type the true number of picks made
  and the "next turn" maths stays right even with a half-updated board.

  How it works without any Python running: hidden helper columns on the tracker
  hold live `COUNTIFS` ranks of the undrafted pool, keyed as `POS|rank`, so the
  assistant pulls the nth-best available with plain `INDEX/MATCH`. A hidden
  `LiveCalc` sheet then turns availability into a *probability*: it models each
  player's realised draft slot as `Normal(ADP, ADP_sd)` and computes, via
  `NORMDIST`, the expected VBD of the best player at each position still on the
  board at your next turn — so a receiver at ADP 45 with your next pick at 25 is
  treated as ~95% safe, not a binary 100%, and a player at ADP 20 gets partial
  credit for possibly falling to you rather than a flat 0%. No `FILTER`,
  `XLOOKUP`, or `MINIFS`-style functions that need the `_xlfn.` prefix and are
  not safe across every spreadsheet app.
- **Metrics** — plain-English glossary of every column: what it means, and how
  to actually use it. Start here.
- **Sources** — every dataset with its pull timestamp. Check this before
  trusting the board on draft day.

## The draft simulator

```bash
python src/simulate.py --slot 5 --sims 50000
```

VBD ranks players by value but is blind to *availability*, which is what
actually decides a draft. The simulator draws each player's realised draft slot
from `Normal(ADP, ADP_stdev)` using FFC's real-draft sample, sorts to get one
plausible draft order, and repeats it thousands of times. From that it derives
each player's probability of surviving to each of your picks, and VONA.

Concretely, from seat 5: at pick 5 the expected WR drop by waiting a round is
**80 VBD** vs 24 for RB and ~0 for QB/TE — take a receiver. By pick 45 the board
has flattened to RB 20 / TE 16 / WR 15, and Josh Allen (96% likely to survive)
reads "can wait".

### Opponent model

`simulation.opponents.model` picks how the other teams draft:

- **`roster_aware`** (default) — a pick-by-pick simulation where each team
  values players by an ADP draw but will not draft a position it has already
  filled, and gets a small nudge toward positions where it still needs a
  starter. This makes every simulated draft a set of *valid rosters*, which
  fixes the availability curves — the old model would happily leave a team with
  five RBs and no WR, "taking" receivers in worlds where nobody would.
- **`adp_only`** — the original independent-draw model. Faster, unbiased in
  aggregate, but internally incoherent.

The nudges (`start_bonus`, `flex_bonus`, `caps`) only break near-ties in ADP;
they never drag a round-10 player into round 2. Every run prints a calibration
line — simulated mean draft slot vs real ADP. Current fit: **corr 0.993,
median error 4.4 picks**, and simulated positional demand (QB17/RB42/WR57/TE13)
tracks the observed ADP demand (QB18/RB44/WR63/TE15).

This is deliberately calibrated to *data*, not to folk wisdom. The model
encodes "teams draft coherent rosters" — true, and checkable against ADP — but
**not** heuristics like "never draft a TE before round 5," which the ADP data
already refutes. The board's whole point is to catch where consensus behaviour
and actual value diverge; baking human draft myths into the opponents would
launder those myths back out as if the model discovered them. See
*A note on psychology* below.

All N teams are modelled, including yours; availability at your pick is read
from the board state right before it. Set your seat in `config/weights.yaml`
under `simulation.draft_slot`, or pass `--slot`.

### A note on psychology

Two mindsets fight during a draft. *Value* thinking says take the best player
regardless of position; *roster* thinking says you cannot start five running
backs. Both are right in moderation and dangerous taken alone — pure value
leaves you with an unstartable team, pure roster-filling makes you reach for a
kicker in round 8.

This system encodes just enough positional sense to be realistic (VBD makes
positions comparable; the FLEX overlay and scarcity strip show when a position
is drying up) without encoding the *myths* that positional thinking breeds. A
belief like "TEs don't matter early" feels like positional sense but is a
data-contradicted heuristic; the tier and VBD math will happily tell you to
take Trey McBride in round 2 if that is where the value is, and you should
listen to it over the folk rule. The opponents are modelled the same way — real
enough to draft valid rosters, not so "human" that they replay draft-Twitter
consensus. Where the numbers and your gut disagree, the numbers have already
accounted for the thing your gut is reacting to.

## Refresh schedule

Timings relative to your draft date:

| When | Command |
|---|---|
| 10–14 days out | `python fantasydraft/run_pipeline.py` (full) |
| 5–7 days out | `python fantasydraft/run_pipeline.py` — post roster-cuts depth charts + fresh ADP |
| 48 hrs out | `python fantasydraft/run_pipeline.py --refresh market` then `--stage features` |
| Night before | Don't re-run. Use the `Notes / Manual Flag` column on the Master Board. |

`injuries_<season>.csv` does not exist on nflverse until the season's first game
reports are published, so a July run skips it — expected, not an error.
Durability still comes from `games_missed_l3y`. Once preseason reports land it
is picked up automatically.

## Out of scope (per spec)

Weather, in-season start/sit, auction dollar values, web UI.
