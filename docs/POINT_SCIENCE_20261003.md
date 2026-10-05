# Henachel POINT — première étude prédictive, 3 octobre 2026

## Décision

**Retenir uniquement max_depth=4 au lieu de 6.** Les 81 features, les autres paramètres et la politique de calibration restent identiques. Le candidat franchit le seuil walk-forward puis la confirmation finale indépendante; aucun réglage de picks modifié.

## Clôture opérationnelle

Branche `astra/audit-remediation` synchronisée avec origin, HEAD `528fd3e8d7a6336817451a233b8b84a30364e5e7`. GitHub Actions `henachel-points-global` #219, identifiant 37131198300 : SUCCESS confirmé pour ce SHA. Remédiation technique terminée. Migration Sheets, scraper et matching non réaudités.

## Protocole et données

Snapshot réel figé : 101 260 lignes, 95 212 saison régulière et 6 048 playoffs, aucun preseason. SHA-256 et provenance dans `input_manifest.json`. Baseline et protocole conservés sans écrasement. Les CSV ont été compressés sans modification des contenus après saturation du disque; le SHA-256 des données décompressées est vérifié. Développement seulement avant le 29 mars 2026; réserve finale non utilisée pour choisir les essais; évaluée une seule fois après verrouillage du candidat profondeur4.

4 fenêtres expanding-window, minimum 160 dates NHL antérieures, fenêtres de 60 dates NHL. Même fonction d’entraînement/calibration que 05, mêmes lignes pour toutes les variantes. Refit par fenêtre (pas chaque jour): mesure du modèle courant avec une cadence expérimentale plus espacée que la production. Modèle et calibrateur voient exclusivement le passé; sélection des colonnes constantes/manquantes sur train seulement. Pondération de classe inchangée.

Entraînement original conservé (régulière + playoffs); métriques séparées. Primaire: log loss régulière; Brier co-critère. Bootstrap apparié par blocs de 7 dates NHL, 10 000 réplications, graine fixe, correction Bonferroni sur 18 comparaisons. Promotion préfixée: amélioration log loss ET Brier dans les quatre folds et borne supérieure de l’intervalle corrigé <0, puis confirmation finale après choix verrouillé. Le candidat profondeur4 est seul à passer et a été verrouillé dans final_choice_locked.json avant confirmation finale.

`02b_ablation_point_model.py` ne mesure pas directement cette baseline: sous-ensemble de features, filtrage historique/TOI et ancienne séparation. Les essais isolés réutilisent donc le fit réel de 05, sans modifier 02b ni le pipeline.

## Baseline hors échantillon

| Population | N | Log loss | Brier | AUC | AP |
|---|---:|---:|---:|---:|---:|
| regular | 46866 | 0.600418 | 0.206099 | 0.676728 | 0.522513 |
| playoffs | 3096 | 0.601343 | 0.206237 | 0.669775 | 0.523663 |
| all | 49962 | 0.600476 | 0.206107 | 0.676201 | 0.522307 |

Saison régulière: prévalence POINT 1+ 34.65%; précision top 10% 62.15%, lift 1.794; précision p>=0,5 60.16%; ECE 10 bins 0.009944. Ces taux ne sont pas des ROI.

Baseline prévalence historique: log loss 0.645302, Brier 0.226460. Baseline fréquence joueur pondérée: 0.622459, 0.210869.

Ces métriques ne reproduisent pas les anciens chiffres documentaires: univers et protocole diffèrent. Aucune comparaison avant/après ne doit confondre ces datasets.

| Fold | Évaluation | N régulier | Log loss | Brier | Calibration |
|---|---|---:|---:|---:|---|
| 0 | 2025-03-31 → 2025-06-06 | 4891 | 0.602972 | 0.207268 | sigmoid |
| 1 | 2025-06-09 → 2025-12-03 | 15300 | 0.599848 | 0.205958 | sigmoid |
| 2 | 2025-12-04 → 2026-02-04 | 17136 | 0.603544 | 0.207350 | sigmoid |
| 3 | 2026-02-05 → 2026-03-28 | 9539 | 0.594410 | 0.203477 | sigmoid |

## Calibration

Sigmoid choisie dans les quatre folds par la sélection temporelle interne raw/sigmoid/isotonic. Aucun choix sur les observations évaluées. La production utilise déjà cette méthode: aucune modification nécessaire.

Probabilités brutes régulières: log loss 0.640762, Brier 0.223605. Après calibration existante: 0.600418, 0.206099. Ce bénéfice est celui de la calibration déjà en place, pas une amélioration créée ici.

| Tranche | N | Proba moyenne | Fréquence observée |
|---|---:|---:|---:|
| 0%–10% | 200 | 8.03% | 10.00% |
| 10%–20% | 5804 | 17.10% | 18.18% |
| 20%–30% | 16127 | 24.88% | 24.31% |
| 30%–40% | 10897 | 34.57% | 34.71% |
| 40%–50% | 6581 | 44.70% | 47.04% |
| 50%–60% | 4581 | 54.57% | 57.19% |
| 60%–70% | 2172 | 64.05% | 63.77% |
| 70%–80% | 470 | 73.58% | 71.49% |
| 80%–90% | 34 | 81.78% | 73.53% |

Au-delà de 80%, seulement 34 observations régulières: aucune conclusion robuste sur ces extrêmes. Aucun cas >=90%.

## Inventaire et ablations

Les 81 définitions figurent dans [l’inventaire versionné](point_science_20261003/feature_inventory_detailed.csv). [Cet inventaire](point_science_20261003/feature_inventory_detailed.csv) donne pour chacune source, temporalité, risque, NaN, statistiques par saison, importance par permutation et corrélations. `feature_spearman.csv` fournit la matrice complète; `redundant_pairs.csv` contient les 29 paires |rho|>=0,95.

Attention: NaN après construction != indisponibilité de la source. Les défauts fixes peuvent masquer le manque de saison précédente ou de contexte. Les indicateurs de retour sont des proxies de participation, pas des blessures confirmées; return_from_absence_flag reste actif après un épisode ancien. La profondeur historique évolue fortement entre les deux saisons.

Importance diagnostique élevée: PP moyen 5, points/match pondérés, TOI retour, référence PP avant absence. Une permutation ne mesure pas une contribution causale et les variables corrélées se remplacent. Les suppressions par groupe testent le système complet, incluant la sélection automatique du calibrateur.

| Variante | Δ log loss | Δ Brier | Folds meilleurs sur les deux | Conclusion |
|---|---:|---:|---:|---|
| depth4 | -0.003302 | -0.001456 | 4/4 | candidat_a_confirmation |
| leaf100 | -0.001536 | -0.000709 | 3/4 | instable |
| plus_drought_relative_pre | +0.000325 | +0.000094 | 1/4 | instable |
| plus_drought_surprise_pre | +0.002598 | +0.000547 | 1/4 | instable |
| plus_pp_share_5_pre | -0.000177 | -0.000204 | 3/4 | instable |
| plus_shots_trend_5_10_pre | +0.000970 | +0.000348 | 2/4 | instable |
| plus_toi_trend_5_10_pre | -0.000678 | -0.000322 | 3/4 | instable |
| without_head_to_head | -0.000354 | -0.000182 | 3/4 | instable |
| without_recent_scoring | -0.000642 | -0.000296 | 3/4 | instable |
| without_return_absence | +0.006099 | +0.000917 | 1/4 | instable |
| without_sample_depth | -0.000718 | -0.000474 | 3/4 | instable |
| without_schedule | +0.001067 | +0.000376 | 2/4 | instable |
| without_season_history | +0.000185 | +0.000159 | 1/4 | instable |
| without_shots | +0.000112 | +0.000027 | 2/4 | instable |
| without_standings | +0.001196 | +0.000371 | 2/4 | instable |
| without_streaks | +0.009940 | +0.000886 | 0/4 | groupe_utile |
| without_team_form | -0.000591 | -0.000291 | 3/4 | instable |
| without_usage | -0.000442 | -0.000233 | 3/4 | preuve_insuffisante |

Delta négatif = meilleur. Intervalles ordinaires et corrigés dans [comparisons.csv](point_science_20261003/comparisons.csv); détail des folds dans [fold_comparisons.csv](point_science_20261003/fold_comparisons.csv).

**Utiles:** groupe streaks, dont le retrait dégrade systématiquement log loss et Brier et passe le critère corrigé de dégradation. **Nuisibles démontrées:** aucune feature établie comme telle. **Neutres démontrées:** aucune; absence de preuve ≠ équivalence. **Instables/preuve insuffisante:** autres suppressions et cinq ajouts. Aucun retrait individuel décidé sur une importance faible.

## Nouvelles features et modèles

Drought relatif: d×p/(1−p), où d est le nombre de matchs précédents sans point et p la fréquence historique pondérée; référence géométrique E[d]=(1−p)/p. Variante surprise: −d×log(1−p). Valeurs non définies p∉]0,1[ laissées NaN. Hypothèse stationnaire/indépendante explicite; aucune règle «le joueur doit marquer». Les deux variantes échouent au critère de stabilité.

Autres ajouts: TOI moyen 5−10, tirs moyens 5−10, part PP5/TOI5 (TOI positif obligatoire). Parité historique/futur et invariance aux résultats courants/futurs testées; aucun ajout retenu. Deux modèles HGB seulement: profondeur 4, ou feuille minimale 100, contre profondeur 6/feuille50 existantes. Paramètres, calibrateur et features restants inchangés; résultats ci-dessus.

## Données manquantes et pistes vérifiées

Les données actuelles couvrent tirs, TOI, TOI PP, buts/passes/points, repos, domicile, séries et contexte standings. Elles ne fournissent pas directement PP1/PP2, lignes stables, coéquipiers simultanés, xG, gardien probable horodaté ou distance de déplacement. TOI PP n’est pas une preuve de PP1.

Deux lectures officielles HTTP200 conservées dans `source_probes.json`: [boxscore NHL](https://api-web.nhle.com/v1/gamecenter/2024021178/boxscore) et [penalty kill NHL par match](https://api.nhle.com/stats/rest/en/team/penaltykill?isAggregate=false&isGame=true&start=0&limit=1&cayenneExp=seasonId=20242025%20and%20gameTypeId=2). Le second expose gameId, gameDate, teamId, ppGoalsAgainst et timesShorthanded: un PK adversaire pré-match est techniquement envisageable. Un échantillon accessible ne démontre pas la couverture historique complète. Aucun entraînement PK effectué dans cette étude.

Priorité suivante: collecter/valider toute la couverture PK, construire taux agrégé des matchs antérieurs (pas moyenne naïve des pourcentages), gérer dénominateur nul/NaN, joindre adversaire, tester parité et invariance puis une nouvelle expérience préenregistrée. Autres pistes: tendance TOI PP avec fenêtre 10 pré-match; production PP/5v5 requiert une décomposition événementielle fiable. Gardien probable/lignes nécessitent une archive disponible avant match. Pas de valeurs reconstruites depuis la composition finale.

## Sélection des picks et limites statistiques

Aucun historique fiable de cotes aligné sur ces observations OOF: ROI, edge et seuils optimaux non évaluables. La précision top10% et les top5/top10 quotidiens (`daily_top_diagnostics.json`) sont des diagnostics prédictifs, pas des portefeuilles de paris. Ne pas imposer 10 picks; conserver la séparation probabilité/décision et les critères existants.

Deux saisons seulement, quatre folds, bootstrap approximatif (dépendance joueur/équipe sur longue durée non entièrement couverte). Calibration du deuxième fold issue des playoffs, évaluation ensuite surtout régulière: changement de distribution important. Première saison sans saison précédente dans la base. Les groupes sont larges et l’absence de signal n’exclut pas une interaction utile. Historique composé de joueurs ayant joué: métriques conditionnelles à la participation, pas évaluation rétrospective des scratches/rosters futurs. Les sources peuvent avoir été corrigées après les matchs; aucune archive point-in-time complète ne permet de quantifier cet effet. La réserve finale a été ouverte une seule fois pour baseline et profondeur4 après choix verrouillé, sans affirmation sur son usage dans les études antérieures. Elle est désormais consommée: ne pas la réutiliser pour tuner les prochaines variantes.

## Modèle BUT: état et plan

`model/04_train_goal_model.py` reste un placeholder qui charge un CSV et écrit un résumé. Il n’entraîne aucun modèle BUT. Ne pas présenter POINT comme substitut.

1. Définir cible buts>=1 et univers joueurs/matchs finalisés; préserver mêmes identifiants, statuts et séparation régulière/playoffs. Tester buts manquants/incohérents, doublons et absence de participation.
2. Figer snapshot et protocole BUT distinct, expanding-window + calibration temporelle + réserve finale. Baselines prévalence et fréquence de buts antérieure; HGB simple comme comparateur initial.
3. Réutiliser features communes pré-match; étudier buts/match, fréquence but, tirs 5/10/20, tirs/60, TOI/PP, conversion buts/tirs avec shrinkage appris sur train seulement, drought buts relatif. Ajouter qualité adversaire/PK si couverture prouvée. PP1/xG/gardien seulement avec sources temporelles fiables.
4. Tests parité historique/futur, modification des résultats futurs sans effet sur train/calibrateur, absences/transferts/rookies; cible BUT séparée dans tous les exports.
5. Évaluer log loss, Brier, AP/AUC, fiabilité par tranches et stabilité; ablations avec correction des comparaisons. Choix figé avant test final, aucune adaptation ROI.
6. Après preuve hors échantillon, calibration BUT et intégration explicite du marché BUT 1+, dans un parcours sandbox distinct; ne pas réutiliser implicitement le contrat POINT.

## Tests, commits et reproduction

190 tests de la suite officielle et 13 tests de recherche passants avec warnings en erreurs (parité, absence de fuite, cas NaN, calibration insensible aux labels futurs; trois tests officiels protègent aussi la reproduction et la restauration des paramètres). 76 fits walk-forward exportés (4 fits supplémentaires réexécutés après interruption d’export pour disque plein): baseline4 + ablations44 + ajouts20 + modèles8. Seul max_depth change dans model/henachel/point.py; régression opérationnelle complète relancée après ce changement. Commit local de cette amélioration, aucun push. Fichiers Finder .DS_Store préexistants laissés intacts.

Résultats et scripts conservés localement sous `outputs/point_science_20261003_528fd3e/` (ignoré par Git). Les dossiers d’expériences sont immuables: les scripts refusent de les écraser. Pour reproduire dans un autre dossier sous outputs, copier uniquement `study.py`, `candidates.py`, `compare.py`, `models.py`, `inventory_details.py`, `report.py`, `test_candidates.py`, `confirm_holdout.py`, `input_features.csv.gz` et `input_manifest.json`; exécuter prepare, baseline, ablations, candidates, compare, models, compare, inventory_details, confirm_holdout puis report dans cet ordre (la répétition du final sert uniquement à reproduire les résultats verrouillés, jamais à retuner). Conserver le même commit et environnement.

Le script versionné scripts/evaluate_point_depth.py reproduit uniquement baseline6/candidat4. Il refuse les sorties existantes, fige les paramètres de référence et restaure les paramètres de production après chaque fit. Utiliser --features outputs/point_science_20261003_528fd3e/input_features.csv.gz --output outputs/reproduction_point_depth; ajouter --final avec un autre dossier de sortie pour reproduire la confirmation déjà consommée. Les autres campagnes négatives restent locales.

Commandes de vérification sans réentraîner:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest outputs/point_science_20261003_528fd3e/test_candidates.py -q -W error
.venv/bin/python outputs/point_science_20261003_528fd3e/compare.py
git diff --check
git status --short --branch
```

Les anciens scripts scientifiques ne sont pas remplacés; cette première campagne est une expérience isolée reproductible. La prochaine campagne doit fixer ses hypothèses avant essais, élargir le recul saisonnier et définir une nouvelle réserve indépendante: celle de cette étude est consommée. Aucun modèle plus complexe ou réglage bookmaker n’est justifié par les résultats actuels.

## Confirmation finale verrouillée

Candidat choisi avant ouverture: profondeur 4. 8 964 observations finales, dont 6 012 régulières (22 dates NHL) et 2 952 playoffs. Deux fits supplémentaires, sans réentraînement sur ces résultats.

| Population | Modèle | Log loss | Brier | AUC | Top 10% |
|---|---|---:|---:|---:|---:|
| regular | baseline | 0.609055 | 0.209835 | 0.671873 | 62.46% |
| regular | depth4 | 0.604826 | 0.208197 | 0.679561 | 61.63% |
| playoffs | baseline | 0.610686 | 0.210012 | 0.654880 | 59.46% |
| playoffs | depth4 | 0.603618 | 0.207380 | 0.665180 | 59.46% |
| all | baseline | 0.609592 | 0.209894 | 0.666609 | 61.65% |
| all | depth4 | 0.604428 | 0.207928 | 0.675050 | 60.87% |

Delta log loss régulier -0.004229, IC95% blocs [-0.005142, -0.003437]. Confirmation: True. Sigmoid retenue pour les deux modèles. ECE walk-forward régulier: 0,009944 → 0,008060; quatre folds sigmoid pour profondeur4 aussi.

Limite: seulement 22 dates régulières, donc peu de blocs temporels indépendants. La précision top10% baisse légèrement malgré les gains de log loss/Brier/AUC. Aucun gain ROI revendiqué. Mesure des probabilités améliorée sur cet échantillon; généralisation future à surveiller.


## Pièces versionnées et reproduction

Les [preuves chiffrées](point_science_20261003/evidence.json) conservent le protocole, le SHA-256 de l’entrée décompressée, les versions, périodes et métriques. La baseline profondeur6 reste inchangée dans ces preuves. Les prédictions individuelles et expériences négatives détaillées sont locales sous `outputs/point_science_20261003_528fd3e/`; aucun CSV historique réel volumineux n’est ajouté à Git.

Le script versionné a reproduit **exactement**, différence maximale 0, les deux séries de 49 962 prédictions walk-forward et de 8 964 prédictions finales. Cette répétition vérifie la reproductibilité du candidat déjà verrouillé; elle ne constitue pas une nouvelle sélection.

```bash
# Entrée figée locale; fournir ce même snapshot pour retrouver les chiffres.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/evaluate_point_depth.py --features outputs/point_science_20261003_528fd3e/input_features.csv.gz --output outputs/reproduce_depth_walk_forward
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/evaluate_point_depth.py --features outputs/point_science_20261003_528fd3e/input_features.csv.gz --output outputs/reproduce_depth_final --final
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests -q -W error
```

Ne pas modifier les paramètres puis relancer `--final` pour optimiser ce jeu déjà consommé. Aucun push ni exécution distante de ce nouveau paramétrage pendant cette étude; #219 valide le commit opérationnel antérieur.
