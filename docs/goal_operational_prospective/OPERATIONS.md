# GOAL Operational V1 — experimental daily execution

`algorithm_v1.json` freezes the refit algorithm, not daily weights. `protocol.json`
was registered before integration. Neither may be retuned using prospective results.
Runtime verifies the frozen code hashes. POINT training, calibration, features and
selection remain unchanged (the shared feature change only handles empty history).

Manual workflow: `goal_research_enabled=true`. Scheduled workflow: repository
variable `HENACHEL_GOAL_RESEARCH_ENABLED=true`. Default is disabled. A GOAL failure
is explicitly reported and must not invalidate POINT. It creates a coverage gap,
not a zero probability. No GOAL output is published to the business Sheets.

Predictions archive operational, frozen-reference and player-smoothed probabilities
before puck drop. PP evidence lacking a verified source timestamp remains UNKNOWN.
Outcomes append separately; corrections never rewrite pregame evidence.

The canonical SQLite archive is restored by branch and hash-verified from each
rolling 90-day artifact. GOAL-enabled runs additionally upload a gzip backup into
the **unpublished draft** release `henachel-prospective-backups`. Asset names
contain the uncompressed SHA256. Existing assets are downloaded and verified,
never replaced. Each upload is downloaded, decompressed and compared byte for
byte. This release has no artifact expiration; owner deletion/repository loss
remain risks, so retain an independent local copy. Recovery from a draft asset
requires authenticated GitHub access; decompress, verify the SHA256 from its
filename, run `scripts/seal_prospective_archive.py`, then explicitly import the
verified archive. No automatic empty replacement is permitted.

The initial `history_import_ref` dispatch input imports an existing branch ledger
only when the destination has no artifact. It preserves settled rows, refuses a
missing/expired source and cannot be combined with bootstrap. Subsequent runs
restore their own branch normally. Production history must not be bootstrapped.

Run offline checks with `bash scripts/validate_local.sh`. Actual activation on main
requires a green opt-in branch run, offline CI and a verified existing main ledger.
Never bootstrap main to bypass a missing canonical history.
