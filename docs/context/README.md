# Pregame context v1 (annotations, never a model)

`refresh_context.py` reads **existing** pregame POINT/GOAL batches. It never fits,
changes probabilities, calls the bookmaker, settles bets or writes Sheets.
`context_baselines`, `context_observations`, `context_views` are immutable tables
in the same prospective SQLite archive. Its existing backups include these tables;
sealing checks their restoration too. `BEGIN IMMEDIATE` serializes local writers.

The earliest archived prediction is retained. The archived `model_rank` is preserved unchanged. Legacy rows without it use
an explicitly labelled probability order (ties game/player), never represented
as the betting-policy rank.
Original payloads, probability, bookmaker evidence and prediction batch references
are retained. Views do not replace the canonical bets ledger or its decisions.

Each observation records game/player/team, schema, actual collection time, source,
source timestamp **type**, freshness, evidence, value and deterministic hash.
Historical statistics use actual retrieval of official final data as evidence,
not a fabricated publication time. Each contributing final game, observed-final
time, statistical window and data-through bound is recorded; target/live games
are refused. No unknown value is zero-filled. Partial statistical values remain null.

Sources: NHL v1 club schedule/boxscore, NHL stats REST game-level team powerplay,
penaltykill, penalties and skater powerplay. Bounded last five completed team games
(regular/playoffs), player present appearances only. PP share uses the same sample.
Missing/truncated reports remain unknown; no season-wide performance is claimed.
Goalie participation in prior games is historical, **not a starting-goalie forecast**.
Season goalie performance, xG/dangerous shots, travel, injuries and current lines
remain UNKNOWN without a verified source. Optional `--enable-pp` reuses RotoWire's
existing parser; its publication timestamp is unverified, so units stay UNKNOWN.
DailyFaceoff is not queried. No source restriction is bypassed.

Rules: only a fresh CONFIRMED OUT/SCRATCH/NOT_PLAYING observation can deactivate a
player. Conflicting confirmed statuses keep the player active for review. PROBABLE,
UNKNOWN and stale evidence cannot cause a certain exclusion. Latest observation
per source/category is explicit; conflicting sources and same-time contradictions
remain visible. Every changed view references its predecessor and observations.
No bonuses, penalties, model reranking or value-policy changes occur.

Workflow: existing daily run adds a fail-soft step when repository variable
`HENACHEL_CONTEXT_ENABLED=true`. Same workflow offers `context_only=true` manual
refresh and hourly cron at minute 17. The same concurrency group serializes both
paths. Intraday runs restore/upload the same canonical history and use the existing
durable backup job. Default window 18 hours, default minimum interval 60 minutes,
configurable via `HENACHEL_CONTEXT_MIN_INTERVAL_MINUTES` (15–1440). UTC execution
and event timestamps govern eligibility; scheduled jobs may be late or skipped by
GitHub concurrency. No eligible archived game means zero source requests. There
is no automatic collection for games without baseline predictions.

Isolated real test (never production writes):
```
python scripts/refresh_context.py --archive COPY_OR_READONLY_SOURCE.sqlite3 \
  --output-dir outputs/context_dryrun_NEW --dry-run --enable-pp
```
The destination must be new. Dry-run uses SQLite backup; the source is read-only.
Reports and views are written in timestamped folders. Unknowns/source failures are
logged; source failures do not affect model outputs. Statistical contextual data
are not new predictive features and no scientific benefit has been established.

The existing frozen GOAL protocol is untouched: its file still specifies a final
window beginning 2026-11-01. The communicated operational tracking date 2026-10-17
is not silently substituted for that preregistered evaluation window in this task.
