# Henachel — modèle NHL POINT 1+

Pipeline de prédiction pré-match et de sélection quotidienne. GitHub reste la source officielle ; la remédiation est préparée sur `astra/audit-remediation`, uniquement en local.

## Installation et validation locale

Python testé : **3.11.9**, environnement POSIX (macOS/Linux). Les verrous de l'historique utilisent `fcntl`.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q -W error
```

Les tests utilisent des données synthétiques, des répertoires temporaires et des mocks. Ils n'appellent pas les services NHL/Unibet/Google et ne lisent aucun credential. `requirements.lock` contient les versions effectivement installées et testées ; `requirements.txt` expose les principales dépendances.

## Fonctionnement

Ordre et commandes : [docs/RUN_ORDER.md](docs/RUN_ORDER.md). Contrats, limites et résultats de remédiation : [docs/REMEDIATION_STATUS.md](docs/REMEDIATION_STATUS.md).

- `model/henachel/features.py` définit les mêmes features pour `01` et `05`. Les statistiques courantes sont exclues ; `passes_moy_10` est construite. Les compteurs équipe sont séparés par saison.
- Le modèle reste `HistGradientBoostingClassifier`, avec les paramètres POINT centralisés dans `henachel/point.py`. Pas de nouvel algorithme ni de nouvelle feature métier.
- `henachel/calibration.py` partage le sigmoid sur **logit(p)** entre évaluation et production. Le choix raw/sigmoid/isotonic utilise seulement une séparation temporelle interne de la fenêtre de calibration. Le test final n'intervient jamais dans ce choix.
- Les PP inconnus restent inconnus. Les standings doivent appartenir à la saison du match et être antérieurs à celui-ci, avec date API vérifiée et âge maximal de trois jours.
- Le matching exige une rencontre et une identité uniques, un marché POINT 1+, des cotes valides et des timestamps vérifiables. Les rejets sont exportés avec leur motif.
- `outputs/history/master_daily_bets_history.csv` est l'historique canonique. Google Sheets est une vue. Le settlement précède la publication ; les relances préservent les résultats et les identifiants.
- Une journée sans match, sans cote ou sans candidat peut produire une sortie vide avec en-têtes et un statut explicite.

**Règle métier conservée :** la sélection privilégie la probabilité du modèle et peut retenir une EV négative. L'exclusion « hot streak » est désormais explicitement optionnelle (`07 --enable-hot-streak-exclude`) ; sa valeur prédictive n'est pas démontrée.

Les artefacts historiques correspondant à Brier ≈ 0.205518, log-loss ≈ 0.599639 et AUC ≈ 0.678702 ne sont pas disponibles ici. Ces nombres restent une référence documentaire, pas des résultats reproduits.

## Publication et exploitation

Aucun test local ne publie dans Google Sheets. Le workflow sécurise ses inputs, sérialise les runs, restaure un historique propre à la branche et authentifie Google uniquement en mémoire. Une absence d'historique restaurable bloque explicitement : ne jamais utiliser `bootstrap_history` pour remplacer un historique existant. Les artifacts GitHub (90 jours) nécessitent une sauvegarde durable indépendante.

Le workflow utilise `scripts/collect_unibet_structured.py` : JSON public Unibet, concordance unique calendrier NHL (équipes + heure exacte), roster récent et timestamp réel de collecte. Le matching strict reste inchangé. Voir `docs/UNIBET_STRUCTURED_CONTRACT.md`. Sur `astra/audit-remediation`, les pushes déclenchent une validation isolée ; toute publication métier Sheets est interdite, seul le contrôle en lecture seule est exécuté.

## Règle projet — ajout de features

Avant tout nouvel ajout métier : présenter les critères existants et les exclusions, demander l'accord de l'utilisateur, puis traiter un seul bloc à la fois avec validation temporelle. Aucune information postérieure au début du match ne peut devenir une feature pré-match.

POINT utilise désormais la profondeur 4, validée en walk-forward et sur holdout ([étude POINT](docs/POINT_SCIENCE_20261003.md)). Le workflow #220 valide ce changement sur `astra/audit-remediation`.

`BUT = research only / not production approved`. `04` est un véritable entraînement BUT de recherche, distinct de POINT. Son premier holdout ne démontre pas de supériorité sur une fréquence joueur lissée : aucune intégration bookmaker/publication BUT n'est activée ([étude BUT et reproduction](docs/GOAL_SCIENCE_20261004.md)).
