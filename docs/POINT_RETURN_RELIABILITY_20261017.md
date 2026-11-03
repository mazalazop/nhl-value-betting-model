# P0 — Fiabilisation des signaux de retour après absence (17 octobre 2026)

## Résultat technique

- Le code historique (`henachel.features`) utilise un compteur cumulatif des épisodes d'absence par joueur qui traverse les saisons.
- Le drapeau `return_from_absence_flag` reste donc vrai pendant toutes les saisons suivantes après un premier épisode; `return_stabilized_flag` peut rester activé bien après le retour.
- Ces variables pilotent une pondération 60/40 ou 80/20 de l'historique POINT et appartiennent aux caractéristiques du modèle.
- **Attention : une absence de dix jours entre matchs observés est un proxy de calendrier, pas la preuve d'une blessure médicalement confirmée.**

## Impact mesuré avant réentraînement

Archive GitHub Actions 17/10/2026 : 256 009 lignes joueur-match.

| Saison | Ancien « retour » actif | Nouveau retour actif | Ancien « stabilisé » | Nouveau stabilisé |
|---|---:|---:|---:|---:|
| 2021-22 | 67,05 % | 10,51 % | 37,47 % | 8,18 % |
| 2022-23 | 93,29 % | 7,20 % | 58,12 % | 5,81 % |
| 2023-24 | 95,51 % | 7,54 % | 57,70 % | 6,44 % |
| 2024-25 | 96,15 % | 7,74 % | 57,86 % | 6,68 % |
| 2025-26 | 95,12 % | 8,14 % | 57,48 % | 6,83 % |
| 2026-27 (partielle) | 94,31 % | 0,30 % | 60,32 % | 0,00 % |

Écarts sur la totalité de l'archive : 208 405 lignes changent de drapeau « retour », 120 729 changent de drapeau « stabilisé », et **120 729 changent de poids historique**. Calcul basé sur les colonnes historiques et la fenêtre 0-2 matchs après l'absence, puis 3-8 matchs de stabilisation sous condition de TOI.

## Mise en œuvre isolée

La référence `model/henachel/features.py` est **inchangée** afin de respecter le verrou d'intégrité GOAL17 (hash déclaré dans `docs/goal_operational_prospective/algorithm_v1.json`).

- `model/henachel/point_return_patch.py` réévalue les drapeaux au niveau joueur + saison, puis les poids et moyennes pondérées pré-match.
- `model/01_build_base_features.py` applique cette correction aux features historiques POINT.
- `model/05_predict_upcoming_games.py` applique la **même** correction sur historique + rencontres futures, pour conserver la parité temporelle.
- Les valeurs du match en cours ne sont pas utilisées pour réparer ses propres features.
- `tests/test_return_window_reliability.py` protège la durée finie des épisodes, les reprises successives et le reset intersaison.
- `tests/test_feature_parity.py` évalue la parité **après** la surcouche POINT.

GitHub Actions [run 39040268137](https://github.com/mazalazop/nhl-value-betting-model/actions/runs/39040268137) : **336 tests réussis**, y compris GOAL17 governance.

## Promotion scientifique : non encore autorisée

Cette branche est une **PR de recherche (draft)** : tests de fonctionnement et de cohérence OK, performance prédictive non encore démontrée.

Avant fusion vers `main` :
1. Reconstruire les données historiques avec les mêmes dates de coupure et les mêmes sources figées.
2. Rejouer le pipeline POINT profondeur=4 historique (baseline), puis POINT profondeur=4 + correctif, mêmes folds de validation expanding-window et mêmes calibrateurs temporels.
3. Comparer log-loss et Brier par fold, calibration, top picks, et spécifiquement sous-groupes après absence vs autres joueurs. Répéter sur saison régulière et playoffs séparément.
4. Quantifier stabilité aux changements d'équipe, aux intersaisons et aux calendriers irréguliers ; contrôler l'absence de fuite temporelle.
5. N'approuver que si les critères de validation hors échantillon sont satisfaits et les probabilités recalibrées. En l'absence de gain, conserver un avertissement structurel en recherche et ne pas changer le modèle validé.
6. Ne jamais modifier `algorithm_v1.json` ou sa signature pour forcer les tests GOAL à passer.

Aucune nouvelle promesse de rentabilité ou de classement amélioré ne découle du décompte de lignes affectées.
