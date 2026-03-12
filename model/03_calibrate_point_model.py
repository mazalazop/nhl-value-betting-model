import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")


def build_summary(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def main() -> None:
    input_candidates = [
        OUTPUTS_DIR / "predictions_validation_point_enrichi_v2.csv",
        OUTPUTS_DIR / "predictions_test_point_enrichi_v2.csv",
    ]

    missing = [str(path) for path in input_candidates if not path.exists()]

    summary = {
        "status": "placeholder",
        "message": (
            "Ce script servira à calibrer les probabilités du modèle POINT "
            "à partir des prédictions enrichies."
        ),
        "expected_inputs": [str(path) for path in input_candidates],
        "missing_inputs": missing,
        "expected_outputs": [
            "outputs/predictions_validation_point_enrichi_calibre_v2.csv",
            "outputs/predictions_test_point_enrichi_calibre_v2.csv",
        ],
        "next_step": (
            "Copier depuis le notebook Colab les blocs de calibration du modèle POINT."
        ),
    }

    if not missing:
        validation_df = pd.read_csv(input_candidates[0])
        test_df = pd.read_csv(input_candidates[1])
        summary["validation_rows_loaded"] = int(len(validation_df))
        summary["test_rows_loaded"] = int(len(test_df))

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUTS_DIR / "03_calibrate_point_model_summary.json"
    summary_path.write_text(build_summary(summary), encoding="utf-8")

    print("03_calibrate_point_model.py")
    print("Statut : placeholder propre créé")
    print(f"Résumé écrit dans : {summary_path}")


if __name__ == "__main__":
    main()
