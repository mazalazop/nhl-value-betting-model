"""Historical feature CLI; shared definitions also used by script 05."""
from henachel.features import *
import argparse

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-pp-coverage', type=float, default=.90)
    parser.add_argument('--pp-coverage-policy', choices=['warn','error'], default='error')
    args = parser.parse_args()
    ensure_directories()

    df_source, pp_summary = charger_source_avec_pp(args.min_pp_coverage, args.pp_coverage_policy)
    standings_df, standings_file_summary = charger_standings(INPUT_TEAM_STANDINGS)

    base_canonique = build_base_canonique(df_source)
    base_features = creer_features_temporelles_v2(base_canonique)
    base_features_context = enrichir_contexte_v2(base_features, standings=standings_df, config={})
    standings_runtime_summary = base_features_context.attrs.get("standings_summary", {})

    base_canonique.to_csv(OUTPUT_BASE_CANONIQUE, index=False)
    base_features.to_csv(OUTPUT_BASE_FEATURES, index=False)
    base_features_context.to_csv(OUTPUT_BASE_FEATURES_CONTEXT, index=False)

    summary = {
        "status": "ok",
        "input_file": str(INPUT_BASE_MATCH_FUSIONNEE),
        "pp_file": str(INPUT_PP_STATS_GAME),
        "standings_file": str(INPUT_TEAM_STANDINGS),
        "outputs": [
            str(OUTPUT_BASE_CANONIQUE),
            str(OUTPUT_BASE_FEATURES),
            str(OUTPUT_BASE_FEATURES_CONTEXT),
        ],
        "rows_base_canonique": int(len(base_canonique)),
        "rows_base_features": int(len(base_features)),
        "rows_base_features_context": int(len(base_features_context)),
        "cols_base_canonique": int(len(base_canonique.columns)),
        "cols_base_features": int(len(base_features.columns)),
        "cols_base_features_context": int(len(base_features_context.columns)),
        "pp_integration": pp_summary,
        "standings_file_integration": standings_file_summary,
        "standings_runtime_integration": standings_runtime_summary,
        "new_feature_checks": {
            "late_season_flag_sum": int(pd.to_numeric(base_features_context.get("late_season_flag"), errors="coerce").fillna(0).sum()),
            "playoff_pressure_non_zero": int((pd.to_numeric(base_features_context.get("playoff_pressure_simple"), errors="coerce").fillna(0) != 0).sum()),
            "hot_streak_excludes": int(pd.to_numeric(base_features_context.get("hard_exclude_hot_streak_pre"), errors="coerce").fillna(0).sum()),
            "return_stabilized_rows": int(pd.to_numeric(base_features_context.get("return_stabilized_flag"), errors="coerce").fillna(0).sum()),
        },
    }
    from henachel.manifest import manifest
    from henachel.point import POINT_PARAMS
    summary["manifest"] = manifest([INPUT_BASE_MATCH_FUSIONNEE, INPUT_PP_STATS_GAME, INPUT_TEAM_STANDINGS], POINT_PARAMS)
    write_summary(summary)

    print("01_build_base_features.py")
    print(f"Input base      : {INPUT_BASE_MATCH_FUSIONNEE}")
    print(f"Input PP        : {INPUT_PP_STATS_GAME}")
    print(f"Input standings : {INPUT_TEAM_STANDINGS}")
    print(f"Output          : {OUTPUT_BASE_CANONIQUE}")
    print(f"Output          : {OUTPUT_BASE_FEATURES}")
    print(f"Output          : {OUTPUT_BASE_FEATURES_CONTEXT}")
    print(f"Summary         : {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
