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

The canonical SQLite archive is restored by branch, hash-verified, and backed up
in each rolling 90-day GitHub artifact. A local copy is recommended. This is NOT
independent permanent storage: an interruption exceeding artifact retention can
lose the remote copy. Restore fails closed on expiration. Before the prospective
window ends, export a verified archive to durable owner-controlled storage; this
is an explicit operational debt, not a claim of permanent GitHub retention.

Run offline checks with `bash scripts/validate_local.sh`. Actual activation on main
requires a green opt-in branch run, offline CI and a verified existing main ledger.
Never bootstrap main to bypass a missing canonical history.
