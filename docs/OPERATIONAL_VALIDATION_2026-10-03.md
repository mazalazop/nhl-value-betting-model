# Validation opérationnelle réelle — 3 octobre 2026

Statut : **BLOQUÉ pour la validation de bout en bout et la mise en production**.
Les intégrations NHL et le traitement local fonctionnent ; les cotes réelles ne
respectent pas le contrat strict et cette branche n'a pas été exécutée à distance.
Aucun tuning, pari, push, merge, accès aux credentials ou écriture Sheets effectué.

## Périmètre et provenance

- Départ : `3458b82`, branche `astra/audit-remediation`, remédiation locale déjà validée.
- Copie isolée : `/private/tmp/henachel-real-validation-n0112i26`, sources Python seulement.
  Ni credentials ni historique de paris de production copiés.
- NHL : API web NHL pour calendrier/rosters/standings/boxscores ; API statistique NHL pour PP.
  Premiers HTTP 200 conservés depuis le 2 octobre ; refresh principal le 3 octobre.
- Historique public NHL réutilisé depuis l'artifact `11236911254`, puis complété avec
  les résultats NHL réels. Aucun historique de paris extrait de cet artifact.
- Cotes : artifact `11235959909`, normalisé le `2026-10-02T15:43:57Z`.
- Artifacts issus du [run 37028419242](https://github.com/mazalazop/nhl-value-betting-model/actions/runs/37028419242),
  branche `goal-model-live-2026-09-29`, SHA `28b954ad1394cc7873e170d7abdd33d5cd7a0126`.
  Ce code distant est différent du code local testé.
- Logs, résumés et empreintes des sources : dossier isolé `evidence/`, notamment
  `executions.jsonl`, `operational_summary.json` et `history_validation.json`.
  Les manifests applicatifs ont un commit nul dans la copie sans `.git` ; les
  empreintes de la copie et le commit local sont enregistrés séparément.

## Résultats

| Bloc | Observation | Limite |
|---|---|---|
| Calendrier | 4 136 matchs, dont 2 813 finalisés ; 1 344 matchs réguliers 2026-27, 84 par équipe pour les 32 équipes | Photographie du 3 octobre |
| Rosters | 766 lignes, 32 équipes, collecte `2026-10-03T07:56:54Z` | Présence au roster ne garantit pas la participation |
| Boxscores/historique | 101 260 lignes, 2 813 matchs ; 173 matchs absents de l'artifact complétés ; 176 boxscores live | Ancien historique réutilisé ; seulement 3 matchs déjà présents recoupés, zéro divergence de points sur ceux-ci |
| PP | 101 258 lignes, couverture 99,998 %, régulière et playoffs ; 2 absences conservées comme inconnues | Aucun zéro fabriqué |
| Standings | 17 344 lignes, dernière date 2 octobre ; couverture historique jointe 94,276 % | Snapshots absents ou dont `api_date` diffère refusés, notamment hors saison régulière |
| Features | 101 260 lignes construites, pas d'échec | Pas de nouvelle feature métier |
| Entraînement/calibration | 92 296 lignes train jusqu'au 28 mars ; 8 964 calibration du 29 mars au 2 octobre ; sigmoid sélectionnée par la règle temporelle existante | Les métriques internes de sélection ne sont pas une mesure indépendante de performance |
| Prédiction | 564 joueurs, 13 matchs du 3 octobre ; probabilités dans [0,1], zéro doublon joueur/match | Historique strictement antérieur au 3 octobre ; pas de résultat futur utilisé |
| Bookmaker | 35 lignes POINT 1+, 5 événements, cotes numériques valides, zéro doublon | Toutes ont équipe joueur vide ; date de match, début UTC et collecte UTC absents ; NHL game ID également absent mais facultatif |
| Matching/picks | 35 rejets explicites, zéro match validé, zéro pick ; deux exécutions sans erreur | Cas positif avec cotes réelles non validé |
| Publication | Deux vues construites et envoyées à un mock, zéro appel réseau | Authentification et schéma réels Henachel non vérifiés |
| Persistance | Deux états locaux indépendants, restauration, reruns, un win et un loss conservés | Conditions de pari explicitement synthétiques `TEST_ONLY`, résultats NHL réels ; stockage GitHub non validé sur cette branche |
| Reprise partielle | Six processus 09 dont une interruption attendue après journal ; reprise avec deux révisions uniques et timestamps conservés | Essai isolé uniquement |

## Bugs corrigés pendant cet essai

1. Contexte saison 2026-27 encore plafonné à 82 : utilisation de 84 pour cette
   saison vérifiée dans le calendrier réel, dans standings et moteur partagé.
   Les anciennes saisons restent à 82 ; une saison ultérieure inconnue est refusée.
   Régressions dans `tests/test_live_contracts.py` ; parité existante conservée.
2. Rerun sans nouveau pari : suppression de la concaténation vide qui provoquait
   un warning pandas avec le ledger réel de test ; aucune modification du résultat.
   Lecture complète des types CSV dans 09 pour éviter le warning sur les données
   historiques hétérogènes. Régression dans `tests/test_history.py`.
3. Notes du résumé 05 actualisées : le roster récent est prioritaire, contrairement
   au texte ancien encore présent dans la sortie.
4. Harness isolé ajouté avec interdiction explicite d'exécuter 08 ou un script
   arbitraire ; trois tests de ses frontières de sécurité, sans réseau.

## Blocages externes précis

Le normaliseur du repository `mazalazop/nhl-unibet-odds-scraper` n'émet pas les
métadonnées obligatoires. La date de normalisation ne prouve pas la fraîcheur de
la collecte. Le parsing accepte des marchés sans équipe joueur. Un adaptateur
local ne peut inventer ces données : le scraper doit fournir la date NHL, le début
UTC, la collecte UTC, l'équipe joueur vérifiée et une identité événement non ambiguë.
Les libellés réels (`CAR Hurricanes`, `WAS Capitals`, etc.) devront également être
testés avec ce contrat complet. Les garde-fous du matching restent inchangés.
Les cotes du 2 octobre ne constituent pas une collecte fraîche du 3 octobre.

La branche distante `astra/audit-remediation` n'existe pas au contrôle effectué.
Le run distant cité a réussi, y compris publication et upload d'état, mais sur une
autre branche. Il ne valide ni notre YAML sur runner ni notre restauration distante.
La syntaxe/configuration locale et les tests runtime sont vérifiés ; concurrence,
restauration, ordre 00/00a/00b/09/00c/01/05/06/07/08 et artifacts sont présents.
Aucun workflow distant n'a été déclenché et aucun code n'a été poussé.

Le secret Google est annoncé côté GitHub uniquement ; aucune session Sheets
connectée n'est disponible ici. Aucun secret n'a été récupéré, lu ou copié.
Le succès de publication distant est un indice de fonctionnement de cet autre
workflow, pas une validation de nos permissions, colonnes ou mises à jour.

## Reproduction

Validation finale : **146 tests globaux passants**, warnings traités comme erreurs.
Les sous-suites de parité/PP, calibration/science, matching, settlement/historique
et publication/runtime sont également relancées par le script ci-dessous.

Validation locale complète, sans réseau :

```bash
bash scripts/validate_local.sh
```

Rejouer les essais dans l'espace isolé existant (le pointeur est dans
`outputs/operational_validation_location.json`, ignoré par Git) :

```bash
.venv/bin/python scripts/run_operational_validation.py features --date 2026-10-03
.venv/bin/python scripts/run_operational_validation.py predict --date 2026-10-03
.venv/bin/python scripts/run_operational_validation.py dry-run --date 2026-10-03
.venv/bin/python scripts/run_operational_validation.py persistence --date 2026-10-03
```

Pour un nouvel espace, utiliser `init --date YYYY-MM-DD`, puis `sources`, `pp`,
`stats`, `standings`, `features`, `predict` avec la même date. `stats` peut récupérer
l'historique complet sans artifact. Le raccourci `history` est propre à cet essai :
il exige les CSV NHL de l'artifact indiqué sous `evidence/remote_artifacts/model/data/raw/`.
`dry-run` exige le JSON réel sous `evidence/remote_artifacts/market/normalized_points_odds.json`.
Ne jamais y déposer de credentials ou d'historique de paris de production.
Le probe contient les endpoints de référence de cette campagne, pas un contrôle
automatiquement actualisé pour toutes les saisons futures.

Avant fusion et optimisation prédictive : corriger le contrat upstream, récupérer
des cotes fraîches, valider un cas positif de bout en bout, puis tester cette branche
sur un workflow isolé avec sandbox Sheets et persistance entre deux runs distants.
