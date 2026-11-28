# Reproduction et limites

Base de recherche : `3f90a4a` ; protocole initial et premiers outils committés sous `5d7d44937aad7fc85e57c12073fcecea0ada4cad` avant les ajustements. Branche `research/henachel-v3`. Aucun fichier opérationnel modifié. `protected_files.json` contient les empreintes des modèles, scripts, configurations, workflows, documentation BUT V2 et archives postgame à cette base. Ces empreintes historiques restent archivées. Depuis la synchronisation du 9 novembre, `verify_delivery.py` contrôle la base opérationnelle courante grâce à `integration_baseline.json` ; le test V3 vérifie les preuves de recherche immuables et les configurations désactivées. Cela permet aux mises à jour quotidiennes indépendantes de main de continuer. Aucun de ces contrôles ne dépend d'un checkout Git profond.

Les dépendances de calcul sont celles de `requirements.lock` ; utiliser un environnement séparé si nécessaire. Les graphiques demandent matplotlib 3.9.4 ; ici installé dans `outputs/v3/plot-libs` sans toucher à l'environnement de production ni aux requirements. Les scripts de calcul n'ont pas besoin des graphiques.

## Entrées figées

- Historique compressé SHA-256 `bfcedf2876dbc81f4ef73fe4d09297095dc5256918f1d99b5c2928da4f33f826`, récupéré en lecture seule dans le snapshot durable `58b08b1cc6741093d0dc266d98b71e4d1ad7cf65b900aa5db87808089b161df2`. `results/history_receipt.json` précise l'asset et le bundle. `retrieve_history.py --receipt` permet de demander cette version, au lieu de la plus récente.
- Postgame : empreintes des résultats courants et révisions dans `postgame_inputs.json` et `postgame_revisions.json`. Pour reproduire le bilan figé, travailler à la base `3f90a4a` ; un main ultérieur peut contenir de nouvelles journées.
- Original prospectif : rapport vérifié provenant du run [41312380543](https://github.com/mazalazop/nhl-value-betting-model/actions/runs/41312380543).
- A01 : lecture du registre restauré vérifié au run [41316586229](https://github.com/mazalazop/nhl-value-betting-model/actions/runs/41316586229), étude A01-20261026-01. Les rapports sont copiés tels quels, sans relancer son évaluation ni modifier les états.
- Réponses NHL : 2 792 enveloppes locales sous `outputs/v3/nhl_pbp`, empreintes dans `nhl_cache_hashes.json`. Le cache contient URL/heure/document et n'est pas redistribué. Une collecte ultérieure peut contenir des corrections officielles : elle n'est alors **pas** une reproduction exacte. Pour reproduire exactement, conserver le cache correspondant aux empreintes. Les sauvegardes existantes n'ont pas été modifiées pour héberger ces nouvelles données.

Les grandes matrices et prédictions restent hors Git sous `outputs/v3`. Leurs empreintes sont dans `results/evidence_manifest.json`. Le dépôt contient agrégats, intervalles, tableaux et figures, pas les données brutes tierces. La disponibilité future de l'API et des assets n'est pas garantie ; les preuves chiffrées livrées restent inspectables sans accès réseau.

## Commandes

Utiliser des dossiers de sortie neufs : les programmes refusent d'écraser une expérience. Les noms ci-dessous illustrent une reproduction ; les résultats livrés ont les chemins exacts inscrits dans leurs manifestes.

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
.venv/bin/python research/v3/retrieve_history.py --receipt research/v3/results/history_receipt.json --output outputs/v3/reproduction-source
.venv/bin/python research/v3/audit.py --history outputs/v3/reproduction-source/history.csv.gz --output outputs/v3/reproduction-audit
.venv/bin/python research/v3/run_models.py --history outputs/v3/reproduction-source/history.csv.gz --output outputs/v3/reproduction-main
.venv/bin/python research/v3/collect_shots.py --history outputs/v3/reproduction-source/history.csv.gz --output outputs/v3/nhl_pbp
.venv/bin/python research/v3/shot_features.py --history outputs/v3/reproduction-source/history.csv.gz --pbp outputs/v3/nhl_pbp --output outputs/v3/reproduction-shots.csv
.venv/bin/python research/v3/run_quality.py --history outputs/v3/reproduction-source/history.csv.gz --quality outputs/v3/reproduction-shots.csv --output outputs/v3/reproduction-quality
.venv/bin/python -m pytest -q -W error
```

Le script courant `run_models.py` utilise directement la correction 128/50 000. Les premières prédictions livrées ont été conservées et leurs intervalles recalculés par `resummarize.py` ; `main.json` est le rapport corrigé faisant foi. Aucun choix de modèle ne dépend des intervalles initiaux. L'ajout ultérieur de `conversion_gap` à la fonction commune ne modifie pas la liste explicite des variables du replay principal.

`publish_results.py` assemble les sorties de cette session et produit les graphiques ; `write_report.py` rend le rapport statistique depuis les agrégats. `describe_conversion.py` produit uniquement des strates descriptives post-hoc, sans test ni sélection. Les fenêtres et listes exactes figurent dans les manifestes pré-fit.

## Incidents techniques conservés

1. Première vérification d'identités : formats entiers/chaînes flottantes équivalentes ; correction de la comparaison, jamais des identifiants.
2. 22 collectes NHL ont échoué au DNS puis réussi lors d'une reprise du cache ; aucune observation manquante n'a été remplacée par zéro.
3. Premier essai qualité : saisons chargées comme chaînes, filtrage entier vide ; correction et garde explicite contre un univers vide.
4. Deuxième essai qualité : ordre différent entre référence et candidat détecté par assertion ; tri commun joueur-match chronologique avant le calcul. Les essais avortés sont conservés localement, sans résultats présentés comme expériences concluantes.
5. La comparaison des sources protège 55 lignes dont les tirs ne concordent pas. Les buts concordent pour les 100 504 lignes ; aucun résultat officiel n'a été réécrit.

## Validation technique

La suite complète initiale a exécuté **456 tests avec succès** (`pytest -q -W error`, 145,57 s). Un test supplémentaire couvre les saisons sous forme de chaînes et l'alignement des références : **12 tests V3 passent** à cette étape. Deux tests supplémentaires vérifient les retries de collecte et le refus d’archiver une réponse vide. La suite finale locale a ensuite réussi **459 tests** avec avertissements traités en erreurs (180,83 s), dont **14 tests V3**. La CI de la PR exécute la suite complète sur la révision livrée ; son lien et son résultat sont indiqués dans la PR.

Les tests V3 couvrent décalage temporel et invariance au futur, imputation uniquement sur entraînement, identités uniques, appariement, géométrie et données incomplètes, déficit de conversion décalé, qualité historique décalée, immutabilité des snapshots, refus d'activation, CLASSEUR distinct et dédupliqué, protection des fichiers originaux. Les tests existants couvrent A01, restauration/sauvegarde et Shadow. Aucun run de production n'a été déclenché pour valider V3 ; aucun candidat n'a été activé.
