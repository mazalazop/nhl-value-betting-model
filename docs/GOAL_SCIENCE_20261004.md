# Modèle BUT — étude du 4 octobre 2026

## Décision

`BUT = research only / not production approved`

**Modèle BUT de recherche construit; promotion opérationnelle refusée par le critère préfixé.** Le holdout est non concluant face à la fréquence joueur lissée. Aucun marché Buteur, pari, sélection ou publication Sheets n’a été activé. Aucun réglage n’est modifié après cette conclusion.

POINT reste figé à profondeur 4: commit `be524f4ba60da07f828e3ba823c9cd564130af42`, poussé sur `astra/audit-remediation`; [workflow #220 SUCCESS](https://github.com/mazalazop/nhl-value-betting-model/actions/runs/37180884954). Ce run confirme POINT et Sheets, **pas** le nouveau modèle BUT.

`04_train_goal_model.py` n’est plus un placeholder: il entraîne et évalue un HGB BUT distinct via `henachel.goal` et `henachel.goal_experiment`. Il nécessite des arguments explicites et n’est pas connecté au workflow POINT. Les paramètres POINT ne sont jamais modifiés par BUT.

## Données et protocole

Même snapshot réel figé de 101 260 participations joueur/match que l’étude POINT: deux saisons, aucun preseason; métriques régulières et playoffs séparées. Pas de suppression silencieuse de lignes. Cible validée: `buts >= 1`, avec contrôle du label `a_marque_un_but`, des identités, des résultats et du statut final. Une passe sans but reste négative pour BUT.

4 fenêtres expanding-window: minimum 160 dates d’entraînement, fenêtres de 60 dates NHL. Chaque entraînement réserve ses 15% dernières dates pour calibration; le calibrateur est choisi sur une subdivision temporelle interne puis ajusté sur cette fenêtre. Aucune calibration sur l’évaluation.

Réserve finale à partir du 29 mars 2026, jamais utilisée pour choisir une feature BUT ou ses paramètres. **Ces dates avaient déjà servi à POINT**: elles ne constituent donc pas un échantillon entièrement vierge entre projets/cibles corrélées. Protocole et addenda conservés dans [les preuves](goal_science_20261004/evidence.json).

Cinq familles préenregistrées, puis PK optionnel soumis à couverture et tests, enfin deux essais de complexité (profondeur 6; feuille minimale 100). Une seule famille supplémentaire pouvait être retenue. Critère de sélection: amélioration log loss ET Brier dans les quatre folds, borne supérieure de l’intervalle corrigé <0. Bootstrap apparié, blocs de 7 dates NHL, 10 000 réplications; Bonferroni sur 8 comparaisons. Pas de split aléatoire.

Après la baseline de développement, **avant ouverture du final**, ajout documenté d’un comparateur joueur plus exigeant: `(hits_but_saison_avant + 10 × prévalence_train)/(matchs_saison_avant + 10)`. La fréquence brute donne parfois 0/1 et constitue un comparateur fragile en log loss. Le poids 10 est fixé sans recherche; le prior utilise exclusivement les dates d’entraînement. L’addendum renforce le critère, sans modifier le modèle ni revenir sur les résultats finaux.

Acceptation finale: log loss/Brier meilleurs que les comparateurs prévalence, fréquence brute et fréquence lissée; IC95% apparié log loss entièrement négatif; ECE <=0,03. Ce critère n’est pas atteint.

## Résultats

Walk-forward: 49 962 observations au total, dont 46 866 régulières. Holdout: 8 964 au total, dont 6 012 régulières sur seulement 22 dates NHL. Résultats réguliers:

| Modèle / période | Log loss | Brier | AUC | AP | Top 10% | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Baseline 15 features / WF | 0.400832 | 0.122043 | 0.679708 | 0.265201 | 32.39% | 0.012012 |
| Candidat 17 features / WF | 0.399666 | 0.121783 | 0.682841 | 0.267185 | 32.24% | 0.011296 |
| Candidat verrouillé / final | 0.401704 | 0.122812 | 0.691095 | 0.256834 | 31.73% | 0.009212 |
| Prévalence / final | 0.429099 | 0.130081 | 0.500000 | 0.153693 | 16.94% | — |
| Fréquence joueur lissée / final | 0.402141 | 0.122933 | 0.680531 | 0.269296 | 32.56% | — |

Contre la fréquence lissée, Δ log loss final = **−0,000437**, IC95% **[−0,006270 ; +0,004574]**; Δ Brier = −0,000121. L’avantage n’est pas établi. L’AP et la précision top10% sont inférieures à ce comparateur. Ne pas revendiquer une meilleure sélection de paris ou une rentabilité.

## Features retenues pour le candidat de recherche

15 features initiales: domicile, nombre de matchs antérieurs, repos; tirs moyens 5/10 et tirs/60; TOI 5/10 et PP5; buts moyens 5/10 et buts/60; fréquences BUT 1+ sur 10 matchs, saison courante avant match, saison précédente.

Famille supplémentaire: `goal_drought_pre` (série sans but, connue avant match et réinitialisée par saison) + `goal_relative_drought_pre = d × p/(1−p)`, avec p fréquence BUT de saison connue avant match. p hors ]0,1[ donne NaN. La référence géométrique est une hypothèse descriptive; pas une règle «le joueur doit marquer». **L’utilité est démontrée en walk-forward pour la famille de deux variables, pas pour le ratio seul.** Le candidat n’est pas promu malgré ce gain de développement.

| Famille / variante | Δ log loss | Δ Brier | Folds meilleurs sur les deux | Seuil passé |
|---|---:|---:|---:|---|
| conversion | +0.000219 | -0.000012 | 2/4 | False |
| usage | -0.000446 | -0.000126 | 3/4 | False |
| context | -0.000657 | -0.000140 | 3/4 | False |
| return | +0.000443 | +0.000102 | 1/4 | False |
| drought | -0.001166 | -0.000261 | 4/4 | True |
| pk | +0.000397 | +0.000092 | 1/4 | False |
| depth6 | +0.003315 | +0.000764 | 0/4 | False |
| leaf100 | -0.001330 | -0.000417 | 2/4 | False |

Les familles de features sont comparées à la baseline 15 features. Les variantes de complexité sont comparées au candidat drought déjà verrouillé, pas directement à la baseline. Conversion/tendance des tirs, évolution TOI/part PP, contexte équipe/adversaire, retour d’absence et PK ne sont pas retenus. Cela ne prouve pas leur inutilité générale.

Le PK officiel couvre les 2 813 matchs (5 626 lignes équipe/match). Calcul par sommes sur 10 matchs antérieurs, même saison; jointure strictement antérieure au jour du match et fraîcheur maximale 30 jours. Zéro opportunité et données absentes restent NaN. Les tests couvrent parité par préfixe historique, exclusion du match courant/futur, saison, fraîcheur et doublons. Les données sont fiables pour cette expérience, mais le gain prédictif n’est pas démontré. Aucun PP1/PP2, gardien probable, ligne ou xG inventé.

## Calibration et limites

Sigmoid sélectionnée dans les quatre fenêtres de la baseline et du candidat. Isotonic sélectionnée dans la fenêtre de calibration antérieure au holdout final, sans consulter ses labels. Final: log loss brute 0,60731, calibrée 0,40170; Brier brut 0,21120, calibré 0,12281. Les probabilités brutes d’un entraînement pondéré ne doivent pas être interprétées comme probabilités de pari.

Fiabilité par tranches disponible dans les trois CSV `*_calibration.csv` versionnés. L’ECE finale est 0,00921, mais une calibration globale correcte ne suffit pas à démontrer un avantage sur un comparateur raisonnable.

Deux saisons, seulement quatre folds et 22 dates régulières finales; dépendances joueur/équipe imparfaitement représentées par le bootstrap. Les statistiques historiques peuvent avoir été corrigées après les matchs; pas d’archive point-in-time exhaustive. Les lignes concernent des joueurs ayant participé: on n’a pas mesuré les scratches/absents de rosters historiques. Les valeurs par défaut héritées des features communes peuvent masquer une faible profondeur historique. Aucune cote historique fiable pour estimer un ROI.

## Tests et fichiers

Tests de séparation BUT/POINT, labels invalides, whitelist anti-fuite, parité historique/futur, invariance aux buts actuels/futurs, exclusion des labels futurs du fit/calibrateur, immutabilité des paramètres POINT, comparaison identique non promue, PK et verrouillage des essais après consommation du holdout.

Le code BUT et le collecteur PK restent des outils de recherche. Les données/OOF compressées restent dans `outputs/goal_science_20261004_be524f4/`, hors Git. Les preuves légères sont versionnées. Le code expérimental était dans le working tree pendant les entraînements; les manifests référencent le commit POINT de départ, et le commit du présent rapport fige le code BUT correspondant.

## Reproduction

Ne pas relancer la sélection dans le dossier déjà terminé: le code refuse les dossiers d’expérience existants et les essais après consommation du holdout. Pour reproduire sans retuner, utiliser un nouveau dossier et le même snapshot/PK; la répétition n’est pas une nouvelle validation indépendante.

```bash
# Remplacer study-dir par un nouveau dossier sous outputs.
.venv/bin/python model/04_train_goal_model.py --features outputs/point_science_20261003_528fd3e/input_features.csv.gz --output outputs/study-dir --phase prepare
# Phases suivantes: baseline, families, pk (avec --pk <pk_games.csv.gz>), models, final.
# Les phases pk et ses tests précèdent le verrouillage modèle.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_goal_model.py tests/test_goal_pk.py -q -W error
```

Le collecteur isolé est `scripts/collect_goal_pk.py --history <snapshot> --output <nouveau_pk.csv.gz>`; il vérifie identités, dates, saisons, pagination et couverture >=95%. Une nouvelle collecte peut inclure des corrections source: comparer les hashes avant de prétendre reproduire les mêmes données.

## Suite nécessaire

Conserver ce résultat non concluant. Ne pas retuner sur les dates finales déjà consommées. Obtenir une validation temporelle indépendante, idéalement prospective sur 2026-27, avec candidate/configuration et comparateur lissé fixés. L’intervalle trop large ne permet pas de conclure à une supériorité; ni l’AUC ni la calibration seule ne lèvent ce blocage scientifique.

L’intégration Buteur Unibet, le matching GOAL 1+, les cotes/edges et le classement top10 BUT sont **non démarrés**, conformément à la condition d’acceptation scientifique. Après validation indépendante, les implémenter en sandbox avec des identifiants et marchés distincts de POINT. Le top5 défenseurs buteurs reste ultérieur. Aucun quota de dix picks ne doit être forcé.


Plan de validation suivante : [validation indépendante et défenseurs](GOAL_NEXT_VALIDATION.md). La suite finale existante a passé 213 tests globaux, dont 23 tests BUT/PK, avec warnings traités comme erreurs.
