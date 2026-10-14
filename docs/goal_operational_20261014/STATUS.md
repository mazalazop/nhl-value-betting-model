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
