# Test fixtures

A frozen board and ingest manifest, so the test suite exercises the code rather
than the internet.

The pipeline pulls from a dozen third-party endpoints that owe us nothing —
FantasySharks 403s datacentre IP ranges outright, so a CI job that re-pulls on
every push goes red for reasons that have nothing to do with the change under
test. A permanently-red suite is worse than none, because people stop reading it.

Live sources are still checked, on a schedule, by `.github/workflows/sources.yml`.
That is a monitoring question, not a pull-request gate.

Regenerate after a deliberate schema change:

    python -m fantasydraft.run_pipeline
    cp data/processed/draft_board.parquet tests/fixtures/data/processed/
    cp data/raw/news_auto.parquet         tests/fixtures/data/raw/
