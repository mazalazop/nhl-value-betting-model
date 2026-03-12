import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FINAL_DIR = DATA_DIR / "final"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")


def build_summary(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def main() -> None:
    features_path = FINAL_DIR / "base_features_context_v2.csv"
    require_file(features_path)

    df = pd.read_csv(features_path)

    summary = {
        "status": "placeholder",
        "message": (
            "Ce script servira à entraîner le modèle POINT à partir de "
            "base_features_context_v2.csv."
        ),
        "input_file": str(features_path),
        "rows_loaded": int(len(df)),
        "columns_loaded": int(len(df.columns)),
        "expected_outputs": [
            "outputs/metrics_modele_point_v2.csv",
            "outputs/metrics_modele_point_enrichi_v2.csv",
            "outputs/predictions_validation_point_v2.csv",
            "outputs/predictions_test_point_v2.csv",
            "outputs/predictions_validation_point_enrichi_v2.csv",
            "outputs/predictions_test_point_enrichi_v2.csv",
        ],
        "next_step": (
            "Copier depuis le notebook Colab les blocs d'entraînement du modèle "
            "POINT baseline et enrichi."
        ),
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUTS_DIR / "02_train_point_model_summary.json"
    summary_path.write_text(build_summary(summary), encoding="utf-8")

    print("02_train_point_model.py")
    print("Statut : placeholder propre créé")
    print(f"Input détecté : {features_path}")
    print(f"Lignes : {len(df)} | Colonnes : {len(df.columns)}")
    print(f"Résumé écrit dans : {summary_path}")


if __name__ == "__main__":
    main()
