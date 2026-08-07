#!/usr/bin/env python
"""End-to-end driver for the draft-board pipeline.

    ./run_pipeline.py                     # full refresh, then build the board
    ./run_pipeline.py --skip-ingest       # rebuild from cached raw data
    ./run_pipeline.py --refresh market    # just re-pull ADP + projections
    ./run_pipeline.py --stage board       # re-run one layer onwards

Each layer is independently re-runnable; the later ones read the parquet files
the earlier ones wrote, so you never have to re-pull nflverse just to retune a
weight in config/weights.yaml.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config  # noqa: E402

STAGES = ["ingest", "features", "project", "board", "simulate", "spreadsheet"]


def main(argv=None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(
        description="Build the fantasy draft board end to end",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--season", type=int, default=cfg["season"])
    ap.add_argument("--history", type=int, default=3)
    ap.add_argument("--skip-ingest", action="store_true",
                    help="use whatever is already in data/raw/")
    ap.add_argument("--refresh", nargs="*", default=None,
                    choices=["all", "nflverse", "market"],
                    help="limit the ingest to a subset of sources")
    ap.add_argument("--stage", choices=STAGES, default="ingest",
                    help="start from this stage (default: ingest)")
    ap.add_argument("--out", default=None, help="explicit xlsx output path")
    ap.add_argument("--slot", type=int, default=None,
                    help="your draft seat (overrides simulation.draft_slot)")
    ap.add_argument("--sims", type=int, default=None,
                    help="number of simulated drafts")
    args = ap.parse_args(argv)

    start = STAGES.index(args.stage)
    todo = STAGES[start:]
    if args.skip_ingest and "ingest" in todo:
        todo.remove("ingest")

    t0 = time.time()
    if "ingest" in todo:
        import ingest
        only = args.refresh or ["all"]
        ingest.main(["--season", str(args.season),
                     "--history", str(args.history), "--only", *only])

    if "features" in todo:
        import features
        features.build(args.season, args.history)

    if "project" in todo:
        import project
        project.project(cfg)

    if "board" in todo:
        import board
        board.build(cfg)

    if "simulate" in todo:
        import simulate
        simulate.build(cfg, slot=args.slot,
                       sims=args.sims or cfg.get("simulation", {}).get("sims"))

    if "spreadsheet" in todo:
        import build_html
        import build_spreadsheet
        print("=== Output: spreadsheet ===")
        build_spreadsheet.build(cfg, args.out)
        print("=== Output: html ===")
        build_html.build(cfg)

    print(f"\ndone in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
