# Prospective PP capture — operational research, not a model change

No training, historical evaluation, PP filter validation, or GOAL promotion is performed.

## Source evidence (2026-10-13)

- Daily Faceoff public line-combinations pages expose PP headings and an update timestamp in web results. Direct unauthenticated collection returned HTTP 403 (Cloudflare). No bypass attempted; no automated adapter approved.
- https://www.rotowire.com/hockey/nhl-lineups.php returned HTTP 200. The parser recovered 60 explicit PP assignments across six teams from the public HTML (full-name anchor titles, team abbreviations, POWER PLAY #1/#2 headings).
- No PP-unit update timestamp was demonstrated in that response. HTTP Date, collection time, and goalie confirmation are NOT source timestamps. Therefore these observations retain `observed_unit`, but operational `pp_unit` is UNKNOWN with `source_timestamp_unverified`. This source is not yet sufficient for a verified PP1 filter.
- Missing players, stale/ambiguous evidence, missing current roster identity, source errors: UNKNOWN. Never infer PP2, PP1, injuries, or historical lineups.

## Local entry point

`python -W error scripts/collect_prospective_pp.py --date YYYY-MM-DD`

Reads the current NHL slate and current rosters, captures the public PP document, and appends evidence to `outputs/prospective/archive.sqlite3`. Optional `--point CSV --goal CSV` consume existing probabilities; no fitting. GOAL CSV requires `proba_goal_1p_calibree`; POINT uses existing production output columns. Both require NHL IDs, full name, team. All records must join the actual upcoming NHL roster/slate. Missing prediction inputs are explicitly reported as partial, not successful end-to-end inference.

`--pp1-only` affects the separate operational ranking only. Raw probabilities/ranking remain intact. No betting recommendation or profitability claim follows from this flag.

## Persistence contract

SQLite batches are committed transactionally, keyed uniquely, protected against UPDATE/DELETE. Exact retries are no-ops even after puck drop; changed retries fail. New pregame inserts check actual local UTC, not only caller-supplied collection time. No postgame results can be inserted as pregame evidence. Outcome batches require final status and known integer goals/points; revisions append separately without changing previous observations.

Archive roles and predictions together with run ID, source-document hash, prediction-file hash, NHL game/player IDs, game date/start, collection/source times, raw observed unit, evidence status and optional pregame TOI/PP TOI. UNKNOWN remains measurable rather than disappearing through filtering. Outcome events retain their own collection timestamp.

This is local crash-safe persistence, not a trusted external timestamp service. Keep a durable off-runner backup; do not rely on expiring GitHub artifacts alone. The workflow now contains an opt-in `capture_prospective_pp` step (or repository variable `HENACHEL_CAPTURE_PROSPECTIVE_PP=true`). It consumes existing POINT output, optionally an existing GOAL research CSV, and validated `06_matched_point_edges.csv`. It does not train a GOAL model. Failure is explicitly warned without changing POINT output. The archive and enable marker travel with the branch-scoped canonical-history artifact; restore refuses an enabled marker without its database. Daily full copies refresh artifact retention, but are NOT a substitute for durable independent backups. This integration has not been executed remotely or enabled on main. No Sheets write was performed.

`python scripts/predict_frozen_goal.py --bundle TRUSTED_LOCAL.joblib --expected-sha256 SHA --future-features PREPARED.csv --output NEW.csv` performs inference only. It requires already prepared, temporally correct GOAL17/C19 features and a trusted, hash-checked bundle with a training cutoff. It refuses current outcomes and late games. It reports the original cutoff and research-only status. It does NOT prepare missing future features or silently substitute a different model. Automatic frozen bundle distribution, daily GOAL feature preparation and outcome capture remain to be connected; no new training was launched.

Matched bookmaker evidence is rechecked for market, event/player/team, timestamps, finite odds and implied probability; rejected rows are counted with reasons. This does not change the selection policy.

## Actual dry run

On 2026-10-13, an initial NHL roster request returned HTTP 429. The new small compressed public cache preserves genuine retrieval times and respects Retry-After (bounded retries, no bypass). The resumed run captured 16 upcoming games / 704 roster players. RotoWire supplied 60 explicit observations. All 704 operational roles remained UNKNOWN: 643 absent observations, 2 identity failures, 59 unverifiable source timestamps. No probabilities were fabricated: zero POINT/GOAL prediction inputs were supplied. Status is `partial_predictions_unavailable`, NOT a successful full end-to-end prediction run.

The append-only archive is local under outputs (not versioned); only the small report is versioned. Initial tests exposed a sandbox physical-core detection warning in an existing synthetic fit test; explicit `OMP_NUM_THREADS=1` resolves the runtime thread budget, without warning suppression or hyperparameter changes.

## Validation checkpoint

`OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q -W error` — 322 passed.
This includes a second independent process reopening the archive, exact replay after puck drop, rejected modifications, partial-batch rollback, legacy history restoration and required prospective-archive preservation. No historical model campaign was executed; existing synthetic model unit tests remain active.

Remaining release gates: an accessible PP-unit source with verifiable update time; current pregame GOAL feature export and explicitly provisioned frozen model; automatic final-result observation capture; full POINT+GOAL+odds dry run; opt-in workflow execution; independent durable archive backup. Do not call this checkpoint a complete operational deployment.
