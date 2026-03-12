import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
FINAL_DIR = DATA_DIR / "final"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def ensure_directories() -> None:
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def build_summary(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def main() -> None:
    ensure_directories()

    summary = {
        "status": "placeholder",
        "message": (
            "Ce script servira à reconstruire les bases propres du projet modèle "
            "à partir des sources validées du notebook Colab."
        ),
        "expected_outputs": [
            "data/final/base_canonique_v2.csv",
            "data/final/base_features_v2.csv",
            "data/final/base_features_context_v2.csv",
        ],
        "next_step": (
            "Copier proprement depuis le notebook Colab les blocs qui produisent "
            "base_canonique_v2.csv, base_features_v2.csv et "
            "base_features_context_v2.csv."
        ),
    }

    summary_path = OUTPUTS_DIR / "01_build_base_features_summary.json"
    summary_path.write_text(build_summary(summary), encoding="utf-8")

    print("01_build_base_features.py")
    print("Statut : placeholder propre créé")
    print(f"Résumé écrit dans : {summary_path}")


if __name__ == "__main__":
    main()
