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

This is local crash-safe persistence, not a trusted external timestamp service. Keep a durable off-runner backup; do not rely on expiring GitHub artifacts alone. Daily workflow integration/restore, frozen GOAL inference, bookmaker evidence and automatic outcome capture are not yet deployed. No production workflow or Sheet has been changed.
