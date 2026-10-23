# Déclenchements et couverture — contrôle ciblé du 23 octobre 2026

## Premier cycle automatique observé

[Run 39949366047](https://github.com/mazalazop/nhl-value-betting-model/actions/runs/39949366047),
événement **schedule**, créé et démarré à **11:54:09 UTC**, révision ea38eb1.
Source, restore, shadow et backup : **tous exécutés et réussis**.
32 tests distants réussis. Source choisie : V1 39928884794.
Sauvegarde draft 421786795, 160 fichiers / 85 069 873 octets,
`remote_restore_verified=true` : restauration complète distante réellement testée.
Snapshot 064b0c9657997e8ebe2cf06348bfb7f71408205b098bb2a949c7fb2b0c0dd351.
Le contrôle indépendant de l'index et des rapports confirme 625 paires sur les
22 et 23 octobre, 425 réglées ; 114 non-participations et 86 en attente pré-match.
La première capture est conservée (SHA256 des prédictions
09c39060f434db0906fcbb88b69dd6e89c2f7d8625e71340f41c9cf83440504d).
Aucune nouvelle capture rétroactive : reprise des dates existantes.

## Créneaux attendus et preuve disponible

Cron installé sur main le 22 octobre à 18:32:56 UTC : `30 6,10,12,16 * * *`.
Aucun créneau restant le 22. Aucun changement ultérieur de ce workflow.

| Créneau UTC du 23 | Observation au contrôle initial de 12:57 UTC |
|---|---|
| 06:30 | Pas de run identifiable à cette heure |
| 10:30 | Pas de run identifiable à cette heure |
| 11:54:09 (effectif) | Un run schedule complet, décrit ci-dessus |
| 12:30 | Aucun nouveau run encore visible (27 minutes de retard possible) |
| 16:30 | Futur, non évalué |

La réponse API et les journaux existants n'archivent pas l'heure nominale de
l'événement schedule. Le run 11:54 est postérieur de 1h24m09 à 10:30 et de
5h24m09 à 06:30 ; **ne pas l'attribuer avec certitude à l'un de ces créneaux**.
Un déclenchement automatique est prouvé, la livraison de chaque créneau ne l'est pas.
GitHub documente retards et omissions possibles sous charge :
https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule
Aucune cause interne de planification ni omission définitive n'est démontrée.

## Pourquoi certains événements ne produisent pas quatre jobs

Cas démontré : [39919064601](https://github.com/mazalazop/nhl-value-betting-model/actions/runs/39919064601)
à 07:38 UTC, workflow_run du V1 39918935326. La source reçoit cet ID explicite ;
les seuls artefacts sont `henachel-canonical-history-f921bd05e68b0374` et
`henachel-context-39918935326`. Aucun `henachel-model-outputs-*` : eligible=false,
restore/shadow/backup skipped. Ce comportement est conforme au filtre : un run
context-only n'est pas un nouveau jeu de prédictions V1. Ce n'est pas un échec
permissions/restore ni une preuve de cycle complet. Les événements context-only
précédents présentent le même motif, notamment 39865174806 le 22 à 22:28 UTC.

Autre cas : V1 39805484632 (22 octobre, fini 16:08:55 UTC) et 39928884794
(23 octobre, fini 09:38:42 UTC), main, success, artefacts modèles présents,
acteur github-actions[bot], événement workflow_dispatch. Le dispatcher utilise
GITHUB_TOKEN avec actions:write. Aucun événement shadow workflow_run correspondant
n'est observé, contrairement aux complétions context-only planifiées.
Le second vient du dispatcher 39928869461 (push main, 09:01:42 UTC) ; ce n'est
pas un lancement humain direct de V1. La branche et le nom du workflow correspondent
aux filtres ; les permissions de lecture des artefacts fonctionnent dans les
reprises manuelles puis planifiée. L'absence se situe donc avant l'évaluation des
jobs shadow, pas dans une condition de job fausse sur un run existant.

La règle anti-récursion de GITHUB_TOKEN est une explication compatible avec cette
chaîne, mais les journaux disponibles ne prouvent pas son application interne à
ces complétions. Ne pas présenter l'hypothèse comme une cause certaine :
https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow
Les exceptions workflow_dispatch/repository_dispatch permettent le lancement V1 ;
elles ne constituent pas à elles seules une preuve de livraison de son événement
workflow_run aval. Aucune preuve d'un dépassement des trois niveaux de chaînage.

## Récupération opérationnelle et limites

Le cron indépendant installé par PR16 découvre les derniers artefacts V1 réussis,
sans dépendre de la livraison workflow_run. Son fonctionnement distant automatique
est maintenant vérifié. Il a repris le V1 39928884794, déjà capturé manuellement
à 10:48 UTC par 39940756196 : **cela ne prouve pas qu'il aurait capturé cette date
à temps sans cette intervention**. V1 avait été généré à 09:36 UTC ; au démarrage
de 11:54 il avait plus de deux heures. Une première capture à ce moment aurait
été refusée par la garde gelée. Le cron est une récupération utile, pas une garantie
de capture dans la fenêtre de fraîcheur. Continuer l'audit externe des journées
éligibles (seuil indépendant 95 %), y compris celles sans événement shadow.

Sérialisation `but-v2-shadow-state`, cancel-in-progress=false conservée. Conditions
main/success/source digne de confiance, gardes temporelles, première paire conservée,
restore isolé contents:write, calcul contents:read, backup contents:write et
restauration vérifiée : inchangés. Variable d'activation true. Aucune erreur de
permissions observée sur ce cycle ; aucune modification de production nécessaire
pour expliquer les runs sautés. Aucun changement du cron sur la base d'une cause
supposée. Si les captures manquées se répètent, évaluer séparément un dispatch
explicite après artefact V1 validé, avec idempotence et tests, sans élargir les
droits du modèle ni assouplir la fraîcheur.

## Couverture

La définition officielle reste WIN/LOSS divisés par toutes les paires capturées.
Cohorte finale du 22 : 425/539 = 78,8497 %. Diagnostics : 114/539 = 21,1503 % de
non-participations corroborées, 0/539 réellement inexpliqué, 425/425 participants
identifiés réglés. Les absents restent officiellement UNRESOLVED.
Le cumul temporaire des deux dates est 425/625 = 68 %, avec 86 matchs-joueurs
encore avant match ; ne pas assimiler cette attente à une anomalie de collecte.

Le gate 98 % est impossible sur la cohorte fixe avec 114 absences définitives,
mais pas mathématiquement impossible sur toute l'étude si les absences suivantes
sont assez rares. Un taux durable d'absence >2 % le rend structurellement
inatteignable, quelle que soit la qualité du collecteur.
Voir [A01-DRAFT-1](COVERAGE_AMENDMENT_01_PROPOSED.md), proposition versionnée
**non appliquée**, pour séparer qualité d'information et participation dans une
éventuelle nouvelle étude, sans réécrire les critères originaux.

Décision humaine restante : valider ou refuser ce principe méthodologique avant
toute implémentation/activation distincte. Aucune promotion ou optimisation.

## Vérification de cette livraison

24 tests locaux ciblés réussis (daily/source/backup/shadow), y compris le contrôle
des empreintes pré-T0. Les 32 tests du run schedule ont réussi sur la révision
réellement exécutée. Cette livraison ajoute uniquement deux documents ; aucun
fichier de code, configuration gelée ou workflow n'est modifié. La proposition
méthodologique demeure une demande de revue, jamais une règle active.

## Actualisation à 15:32 UTC

Toujours un seul événement schedule shadow visible : celui de 11:54. Le créneau
12:30 n'a pas de run identifiable ; 16:30 reste futur.

Deux événements workflow_run supplémentaires sont réellement observés :
- [39962108582](https://github.com/mazalazop/nhl-value-betting-model/actions/runs/39962108582)
  à 13:32:48 : source success, restore/shadow/backup skipped (context-only).
- [39971868349](https://github.com/mazalazop/nhl-value-betting-model/actions/runs/39971868349)
  à 14:42:05 : **source, restore, shadow, backup tous success**. V1 source
  39966909002, événement schedule, vrai pipeline modèle exécuté. Nouvelle sauvegarde
  draft 421959083, 180 fichiers / 85 351 721 octets, restauration intégrale vérifiée,
  snapshot 71c361bad231cf6eb48f9ef0abcaaf14775112a420acab2454fd91f7fe63cfa4.

La chaîne native schedule V1 → workflow_run shadow fonctionne donc réellement.
Le problème observé de livraison aval est circonscrit aux exemples de V1 dispatchés
par GITHUB_TOKEN, sans prétendre prouver une cause interne non exposée par GitHub.
Aucun changement de déclenchement n'est appliqué sans anomalie de configuration
établie ; les deux voies automatiques existantes ont maintenant été exécutées.
