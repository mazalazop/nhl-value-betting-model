# Revue et intégration contrôlée — 21 octobre 2026

Ce document complète et remplace les instructions d'activation initiales de
SHADOW.md : le nouvel exécuteur prépare directement les entrées depuis un run
V1 réussi. Aucun workflow de production n'a été déclenché ni modifié, aucune PR
fusionnée, aucune variable d'activation créée.

## État revérifié

PR9 → main; PR10 → branche PR9; PR11 → branche PR10; PR12 → branche PR11.
Les quatre sont ouvertes; les vérifications initiales étaient vertes et GitHub
les déclarait MERGEABLE/CLEAN. Lors de la reprise, main a avancé à `7a52d24`
avec uniquement des archives postgame et leur résumé. Ces archives doivent
être conservées exactement. Les branches sont synchronisées avec cette base
avant livraison; voir les CI des derniers commits pour l'état actualisé.

Ordre de revue/intégration : 9, 10, 11, 12. Après chaque fusion **autorisée**,
retargeter la PR suivante vers main et attendre sa CI recalculée. Éviter de
supprimer les branches de base avant retarget. Aucun contournement des règles
GitHub ni fusion automatique. La préparation des branches ne fusionne pas les
PR dans main.

## Revue fonctionnelle et corrections

PR9 : aucune écriture sur les modèles/picks POINT ou V1. Les CSV CANONICAL et
les trois groupes restent distincts. Corrections supplémentaires : suppression
du repli dangereux sur un patronyme différent; contrôle ID/état boxscore;
play-by-play partiel détecté par comparaison des tirs officiels; coordonnées
manquantes laissées inconnues; doublon de bet_id contradictoire refusé.
Chaque ancienne vue résultat/résumé est sauvegardée par hash sous `revisions/`
avant mise à jour; les réponses brutes sont dédupliquées par contenu et reliées
par hash aux lignes horodatées. Le glob longitudinal non récursif ne compte pas
ces sauvegardes deux fois. Aucun ancien résultat du dépôt n'a été recalculé ou
modifié pendant la revue. Un nom ambigu reste non résolu, jamais perdant.

PR10 : algorithmes/configuration V2 conservés; labels entiers non négatifs,
whitelist, fit/calibration strictement antérieurs, imputation train-only et
tests par préfixe. Pas de nouveau signal gardien/PP1/xG. Risque résiduel :
historique corrigé non point-in-time, déjà expliqué par le rapport initial.

PR11 : replay historique conservé avec ses preuves et hashes. Aucun retuning,
aucune nouvelle performance prédictive revendiquée. Les intervalles exploratoires
ne sont pas les seuils confirmatoires du futur protocole.

PR12 : univers V1/V2 désormais strictement identique (pas seulement inclusion),
contrôle de métadonnées et fraîcheur V1/prix ≤2h au début du fit. Participation
TOI >0 exigée pour solder; source NHL exacte du match et horodatage vérifiés.
Correction importante du biais de Top10 : sélection figée avant match,
les absents/non résolus ne sont plus remplacés par les rangs suivants.
Pas d'IC de blocs trompeur sur une seule journée. Rapport Markdown/CSV/JSON
incluant calibration, couverture et simulation Top10 distincte pour chaque
modèle seulement sur les cotes pré-match archivées.

## Raccordement préparé, désactivé

`but-v2-shadow.yml` écoute la fin du workflow global, mais son job exige
`HENACHEL_BUT_V2_SHADOW_ENABLED=true` ET main. Variable absente = aucune exécution,
y compris manuelle. La revue n'a pas modifié cette variable.

Le staging vérifie run global main terminé avec succès, bon chemin de workflow,
origine non-PR, un seul artefact outputs non expiré; il télécharge en lecture
seule. Aucun modèle exécutable reçu n'est désérialisé; le code exécuté vient du
checkout revu. Le jeton n'a que contents:read/actions:read, sans secret Sheets.

L'artefact réel `39392942670` a été téléchargé et adapté localement : 495 joueurs,
ID/match/équipes identiques, calendrier vérifié, cotes GOAL 1+ liées aux IDs.
Les fichiers source ont été hashés, sans réentraîner V1 ni produire de capture
shadow. Cet artefact du 20 octobre est désormais historique : impossible de
l'utiliser pour une émission prospective antidatée.

Une seule paire complète est retenue par jour; autres runs et reprises conservent
la première. Un retard après coup d'envoi produit un manque explicite. Une
panne capture n'empêche pas de tenter le bilan des anciennes captures, et le
registre est téléversé même après cet échec. Concurrence globale sans annulation
pour éviter des écritures concurrentes du registre. Premier démarrage vide
uniquement avec `initialize=true` explicite; état expiré ou corrompu = arrêt,
pas de réinitialisation silencieuse.

État isolé `but-v2-shadow-state` : captures, bundles V2, features, sources de
training compressées/dédupliquées, réponses officielles versionnées, bilans et
manifest d'intégrité. Aucun push dans le dépôt. Sources brutes et training ne
sont jamais publiés dans les picks officiels. Les résultats sont recollectés
pendant sept jours; corrections plus tardives à traiter avant décision finale.

## Conditions d'activation à autoriser

1. Revue et fusion des PR dans l'ordre, avec retarget/CI entre chaque étape.
2. Valider [le protocole](PROSPECTIVE_PROTOCOL.md), T0, les gains minimaux et la
   fenêtre confirmatoire avant toute capture. Premier bilan J30, pas de promotion.
3. Vérifier que le run quotidien V1 produit réellement GOAL17 et termine avant
   le premier match, dans la fenêtre de fraîcheur. Succès global seul insuffisant.
4. Prévoir la capacité de stockage et une sauvegarde durable indépendante des
   artefacts Actions. La rétention 90 jours est renouvelée par les snapshots
   cumulatifs, mais un arrêt prolongé peut faire expirer la dernière copie.
   Les snapshots training quotidiens peuvent représenter plusieurs Go sur 180 jours.
5. Après accord explicite, définir la variable true et effectuer le premier
   dispatch shadow sur un **nouveau** run V1 admissible, `initialize=true`.
   Ensuite les fins de runs globaux alimentent le shadow automatiquement.
6. Examiner un premier run réel, sa restauration et son rapport à J+1 avant
   de considérer le fonctionnement distant validé. Ce run n'a pas été exécuté ici.

Arrêt : remettre la variable à false; cela ne touche ni POINT ni V1. Sauvegarder
l'artefact shadow avant expiration, conserver toutes les captures. Toute dérive
du protocole/code gelé demande une revue et un nouveau cycle explicite.

Risques résiduels : latence Actions/collecte, données NHL corrigées après sept
jours, sources contextuelles partielles, coût de stockage, joueurs projetés
absents nombreux qui peuvent empêcher d'atteindre les seuils de couverture.
Les seuils d'effectifs sont des planchers; les IC et l'audit des journées manquées
restent obligatoires. Aucun workflow distant shadow ni gain prospectif revendiqué.
