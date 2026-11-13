# Recherche POINT — backtest apparié du correctif de retour après absence

**Date:** 2026-10-17. **Statut:** NON VALIDÉ POUR PRODUCTION. Ne pas fusionner la PR n°6 sur la seule base des tests fonctionnels.

## Protocole

- Source réelle : artefact GitHub Actions **39021431405**, `data/final/base_features_context_v2.csv` au 17 octobre 2026.
- Comparaison **POINT profondeur 4**, mêmes 81 features et mêmes lignes, seule la transformation des signaux de retour change.
- 243 481 lignes joueur-match dans l'historique de calibration/test avant le 29 mars 2026.
- Quatre fenêtres expanding-window : 2025-03-31 au 2025-06-06 ; 2025-10-07 au 2025-12-03 ; 2025-12-04 au 2026-02-04 ; 2026-02-05 au 2026-03-28.
- Apprentissage avant chaque fenêtre, split fit/calibration chronologique 85/15, poids des classes comme le modèle opérationnel, sélection du calibrateur raw/sigmoid/isotonic **sur les seules dates pré-test**.
- Réplication locale de l'architecture HENACHEL (HistGradientBoostingClassifier, profondeur 4, 300 itérations, learning_rate 0,05, min_samples_leaf 50), non une exécution GitHub Actions de la science officielle.
- IC95 % de la différence des pertes corrigé − ancien par bootstrap apparié de 1 500 échantillons de blocs de sept jours.

## Résultats principaux : 46 866 matchs-joueurs de saison régulière

| Critère | Baseline | Correctif | Delta correctif − baseline |
|---|---:|---:|---:|
| Log-loss | 0,5926366 | 0,5925816 | −0,0000550 |
| Brier | 0,2028313 | 0,2027841 | −0,0000472 |
| AUC | 0,6909853 | 0,6908634 | −0,0001219 |
| Hit Top10 quotidien (167 dates) | 69,16 % | 68,44 % | **−0,72 point** |

**IC95 % :** delta log-loss [−0,000427 ; +0,000309], delta Brier [−0,000215 ; +0,000112]. Les deux contiennent zéro. Top10 : 36 journées améliorées, 40 détériorées, 91 inchangées. IC95 % du delta de réussite Top10 : [−1,98 ; +0,54] points.

| Fenêtre | Log-loss baseline | Log-loss correctif | Brier baseline | Brier correctif |
|---|---:|---:|---:|---:|
| 2025-03-31 — 2025-06-06 | 0,595042 | 0,595023 | 0,203928 | 0,203923 |
| 2025-10-07 — 2025-12-03 | 0,591511 | 0,591515 | 0,202328 | 0,202274 |
| 2025-12-04 — 2026-02-04 | 0,595935 | 0,595770 | 0,204286 | 0,204185 |
| 2026-02-05 — 2026-03-28 | 0,587282 | 0,587312 | 0,200463 | 0,200501 |

## Segments

- **0–2 matchs après absence (3 824 observations)** : log-loss améliorée de 0,001161, Brier de 0,000510.
- **3–8 matchs après absence avec stabilisation (3 305)** : log-loss détériorée de 0,000079 ; Brier de 0,000105.
- **Autres situations (39 737)** : log-loss détériorée de 0,000040 ; Brier amélioré de 0,000015.

Les segments sont exploratoires et sélectionnés en fonction de l'hypothèse de correction. Le gain 0–2 matchs doit être **confirmé sur des dates réellement nouvelles**, pas utilisé pour reparamétrer les seuils a posteriori.

## Qualité / portée de la correction

Ancien drapeau retour actif sur 89,15 % des lignes (jusqu'au 28 mars 2026) ; après correction 8,27 %. Ancien statut stabilisé 53,48 % ; nouveau 6,90 %. **113 400 pondérations historiques** diffèrent sur cette population, avec comparaison numérique tolérante (pas égalité exacte float32/float64).

Le correctif du code partagé `model/henachel/features.py` a été abandonné : **le hash GOAL17 reste inchangé**. L'overlay indépendant `model/henachel/point_return_patch.py` est appliqué par 01 et 05. **336 tests GitHub Actions passent** (run `39040268137` et plus récent après ajustements documentaires).

## Décision et prochaines preuves

**NE PAS promouvoir en production.** L'amélioration prédictive moyenne est négligeable et compatible avec zéro ; légère dégradation du Top10. Garder la PR6 en draft. Des études complémentaires pourront porter sur les 0–2 premiers matchs, avec nouvelles périodes de validation ; ne pas ajuster les 81 features POINT ni le modèle GOAL17 sur ce jeu de test déjà observé.

Limites : source historique précalculée et non recertifiée au niveau de chaque collecte, pas d'historique de vraies cotes pré-match pour ROI, backtest de réplication sans holdout encore vierge. Une absence correspond à un **écart de calendrier joueur**, pas à une blessure médicale prouvée.
