# Shadow isolé — préparation, sans activation

Le candidat logistique est fixé avant les résultats du replay. Aucune
publication Sheets, modification V1 ou recommandation de mise.

Capture :
```
.venv/bin/python scripts/but_v2_shadow.py capture --history history.csv \
  --slate slate.csv --v1 v1.csv --output outputs/shadow/capture-unique
```

`history.csv` : snapshot final avec colonnes brutes du module BUT V2, dates
strictement antérieures au jour cible. `slate.csv` : univers pré-match avec
ID NHL joueur/match, équipe, saison, domicile, date, `start_time_utc`,
`status` FUT/PRE, `schedule_state` OK; buts/tirs/TOI/PP du match inconnus.
`v1.csv` : sortie réellement générée par V1, mêmes IDs, `probability_goal`,
`generated_at_utc`, `start_time_utc`. L'univers V1 doit couvrir V2.
Horodatages obligatoirement avec fuseau, mise au jeu future au début et à la
fin du fit. Aucune option CLI pour antidater la capture.

Cotes optionnelles sur les lignes V1 : `odds_decimal`, `odds_captured_at_utc`,
`odds_source`, `odds_market=GOAL_1_PLUS`; prix observé avant génération et match.
Une cote sans provenance est refusée; une cote absente reste inconnue.
Le workflow de production n'émet pas automatiquement ce nouveau contrat.
Il faut préparer cet artefact explicitement avant raccordement.

Sortie nouvelle et exclusive : predictions.csv, features.csv.gz,
model.joblib, manifest.json (hashes fichiers, modèle, paramètres, cutoff,
identifiant de prédiction, source capturée, horodatage, classement, statut).
L'ancien dossier ne peut pas être écrasé. Les nouveaux bilans ne modifient
pas les captures. `CANONICAL` est ici explicitement limité au scope
`BUT_V2_SHADOW`, jamais joint au registre de paris officiel.
Les contextes PP1/gardien/xG absents sont NOT_AVAILABLE, ancienneté inconnue.
Le TOI moyen ne devient pas un rôle PP1 confirmé.

Bilan :
```
.venv/bin/python scripts/but_v2_shadow.py report --capture outputs/shadow/capture-unique \
  --outcomes outcomes.csv --output outputs/shadow/report-unique
```

Résultats : ID match/joueur, status OFF/FINAL, buts entiers, URL `source`
officielle NHL, `observed_at_utc`. Résultats absents/non finaux restent
UNRESOLVED; pas de perte imputée au joueur absent. Pas de VOID sans règle
bookmaker et preuve explicite. Bilan sur paires identiques, Brier/log-loss,
calibration, Top5/10, intervalles appariés. Si cotes réellement archivées :
probabilité implicite brute (non corrigée de marge), edge moyen, simulation
à une unité sur toutes les lignes cotées et IC par blocs de dates. Aucun
pari réellement effectué ni règle de value sélectionnée après coup.
Un seul jour ne permet pas un intervalle d'incertitude fiable.

`but-v2-shadow.yml` est manuel et réutilisable (`workflow_call`) pour un appel
quotidien ultérieur. Lecture seule du dépôt et des Actions, aucun secret
Sheets, aucun push, aucune programmation activée. Il télécharge l'artefact
`but-v2-inputs` d'un run explicite. Mode capture : history.csv/slate.csv/v1.csv;
mode report : capture/{predictions.csv,manifest.json} et outcomes.csv.
Les outputs reçoivent leur propre nom d'artefact et une rétention de 90 jours.
Une sauvegarde durable supplémentaire sera nécessaire avant suivi long terme.

Validation exécutée : capture → modèle figé → rapprochement officiel → bilan
sur fixtures synthétiques, rejet des matchs commencés, résultats futurs,
V1 post-match, fuseaux manquants, écrasement et altération de capture.
Ce test technique n'est pas une émission prospective réelle. Aucune exécution
Actions shadow ni collecte de nouvelles cotes revendiquée.

Avant activation humaine : revue/merge des PR, alimentation du contrat depuis
les sources rafraîchies, sauvegarde durable, raccordement quotidien au
workflow réutilisable, puis accumulation prospective. Le runtime shadow doit
échouer indépendamment de POINT/V1. Retour arrière : désactiver l'appel shadow,
conserver ses archives, retirer la nouvelle implémentation si souhaité.

## Révision du 21 octobre 2026

La préparation initiale est remplacée pour l'activation par
[CONTROLLED_INTEGRATION.md](CONTROLLED_INTEGRATION.md) et le
[protocole prospectif](PROSPECTIVE_PROTOCOL.md). Le workflow est désormais
alimenté par un artefact du run global vérifié et reste désactivé par variable.
Les commandes capture/report autonomes restent disponibles; les résultats
exigent désormais `toi_min` et `team_player_match`. Le Top10 est figé avant
settlement. La CLI finale `scripts/assess_but_v2_prospective.py` n'autorise aucune
promotion et refuse un bilan avant la fin de la fenêtre fixe.
