# Daily GOAL operational research

`GOAL17_OPERATIONAL_REFIT` is `research_only_operational`, not prospectively production-approved. POINT parameters/features and betting policy are unchanged.

Daily refit uses final regular/playoff rows strictly before NHL target date, exactly GOAL17's 17 features and existing fixed HGB parameters. The existing temporal calibration selector is reused unchanged; training/calibration boundaries and retained columns are recorded. The frozen February 2024 bundle remains a diagnostic reference, never the daily scorer selector.

After the normal NHL/PP/standings/01 refresh:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -W error scripts/predict_goal_daily.py --target-date YYYY-MM-DD --top 10
.venv/bin/python scripts/collect_unibet_structured.py
.venv/bin/python scripts/rank_goal_odds.py
.venv/bin/python -W error scripts/collect_prospective_pp.py --date YYYY-MM-DD --archive outputs/history/prospective.sqlite3 --point outputs/predictions_upcoming_point_enrichi_calibre_v2.csv --goal outputs/predictions_upcoming_goal_research.csv --goal-odds outputs/structured_bookmaker/normalized_goals_odds.json
.venv/bin/python scripts/seal_prospective_archive.py outputs/history/prospective.sqlite3
```

`goal_probability_top.csv` is a probability ranking, not ten recommended bets. `goal_value_ranking.csv` joins verified Unibet group 4266, match excluding shootout, individual 1+ only. `goal_value_shortlist.csv` retains positive edge/EV as an experimental operational rule; no historical ROI claim. Missing odds leave probability rankings available.

`--pp-status PP1` requires explicit `VERIFIED_FRESH` evidence. Current RotoWire observations lack source update timestamps and remain UNKNOWN. The prospective JSON ranking records PP evidence separately, without changing model probabilities.

Opt-in `capture_prospective_pp=true` on this research branch runs refit, goal matching and archive. Keep `publish_to_sheet=false`. First branch run requires explicit `bootstrap_history=true`; later runs restore their own branch's archive. SQLite is append-only; final results/corrections append separate observations. Hash manifest plus backup/restore check precedes artifact upload; restore rejects a mismatching checksum. GitHub artifacts last 90 days: an additional long-term external backup remains necessary before expiry. No production approval, automated real wager, or main merge.

The incremental `refresh_goal_slate.py` helper reuses an existing canonical artifact and only refreshes recent/missing final games. Its weekly schedule adapter explicitly uses NHL day headers rather than UTC calendar dates. It does not replace full refresh verification. Unknown PP stays unknown.

## Completed operational validation

- Local suite: 352 tests passed with warnings as errors; full validation script and dependency check also passed before the final isolated slate-guard test.
- GitHub runs [#243](https://github.com/mazalazop/nhl-value-betting-model/actions/runs/38565001793) and [#244](https://github.com/mazalazop/nhl-value-betting-model/actions/runs/38567838382) succeeded. #244 tested code `356f249` without bootstrap. Sheets writes were skipped.
- October 14 dry run: 104,284 historical observations before cutoff; 2 upcoming games, 85 GOAL probabilities, 64 matched goal odds. Local/remote probabilities agree exactly. Value counts depend on the timestamped odds snapshot, not probability changes.
- Remote archive grew from 2 to 4 batches; every old batch was byte-identical. Canonical ledger stayed at 10 unique bets. Hashes and restore checks passed.
- October 15 observation: 72 actual skater appearances, of which 70 were covered by the prior scored roster. Appended 70 outcomes locally; rerun added zero; 15 scored roster members without returned stats remain unknown. Two actual participants were outside the captured roster universe: no retrospective predictions were manufactured. Rosters do not certify match participation.
- All 85 operational PP annotations remain UNKNOWN. No PP1 freshness evidence or predictive approval is claimed.
- Scheduled GitHub workflows run from the default branch; this unmerged research branch requires daily opt-in dispatch. The next dispatch appends refreshed outcomes to its restored remote archive. The local October 15 outcome append is not claimed as already uploaded to that remote archive.

Compact evidence and hashes: `live_validation.json`, `remote_validation.json`, `outcome_validation.json`, `scored_slate_outcomes.json`. Large datasets, caches, model bundles and archives remain outside Git. No model tuning, main merge or production Sheets publication occurred.
