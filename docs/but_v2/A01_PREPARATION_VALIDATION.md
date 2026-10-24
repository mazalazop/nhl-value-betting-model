# Livraison préparatoire A01 — 23 octobre 2026

PR #17 révisée. Aucun workflow modifié ; étude originale et candidat inchangés.
Le module indépendant classe les observations et le CLI produit un audit séparé
avec preuves horodatées, empreintes et historique des révisions. La configuration
reste DRAFT_NOT_ACTIVATED, enabled=false, t0_utc=null ; toute tentative d'activation
par modification de ces champs est refusée par le préparateur.

## Vérifications exécutées

- 21 tests A01 : absence corroborée, sources vides/tronquées, buts manquants,
  TOI nul/inconnu, report, contradiction, activité play-by-play incompatible avec
  absence, correction finale, retour arrière temporel, doublon, refus d'activation,
  préservation des inputs, refus d'écrasement et d'écriture dans un registre original.
- Suite existante et première version A01 : 401 tests locaux réussis avant l'ajout
  du dernier contrôle play-by-play ; les 21 tests A01 finaux passent ensuite.
  La CI du dernier commit de la PR donne le résultat de la suite complète finale.
- Les 12 SHA256 de PRE_T0_REGISTRATION.json correspondent exactement aux fichiers.
- Le registre original restauré contient toujours les 26 fichiers de son manifeste,
  tous vérifiés ; aucune mutation. Prédictions originales :
  09c39060f434db0906fcbb88b69dd6e89c2f7d8625e71340f41c9cf83440504d.
- Le diff contre main n'inclut ni archives originales, ni modèle, ni workflow,
  ni configuration/protocole gelés. Les évolutions postgame déjà présentes sur main
  ont été intégrées sans réécriture dans la branche de revue.

## Replay exploratoire des sources déjà examinées

Sources officielles horodatées du 23 octobre, cohortes antérieures à tout futur T0'.
Résultat du CLI final : 539 évaluables, 425 participants réglés, 114 non-participations
corroborées, 0 participant à résultat manquant, 0 participation inconnue, 0 report.
Couverture originale inchangée : 78,8497 %. Non-participants : 21,1503 %.
Couverture d'information A01 : 100 %. Disponibilité parmi participants identifiés :
100 %. Les deux booléens diagnostiques à 98 % passent, **aucune validation
confirmatoire ni promotion n'est autorisée**. Ces données restent exploratoires.

## Automatisation : preuve exacte et point manquant

Dernier contrôle de cette préparation : 23 octobre, 16:09 UTC.
- [39949366047](https://github.com/mazalazop/nhl-value-betting-model/actions/runs/39949366047)
  schedule 11:54 : quatre jobs réussis, restauration complète vérifiée.
- [39971868349](https://github.com/mazalazop/nhl-value-betting-model/actions/runs/39971868349)
  workflow_run 14:42 : quatre jobs réussis, sauvegarde draft 421959083 intégralement
  restaurée. Les captures existantes ont été conservées.
- Première capture du 22 : [39770889264](https://github.com/mazalazop/nhl-value-betting-model/actions/runs/39770889264),
  workflow_dispatch manuel ; première capture du 23 :
  [39940756196](https://github.com/mazalazop/nhl-value-betting-model/actions/runs/39940756196),
  workflow_dispatch manuel. Leur reprise automatique ne suffit pas à prouver le
  premier enregistrement automatique d'une nouvelle journée.

Aucune troisième date observable à ce contrôle. **Vérification encore incomplète**
pour la première capture d'une journée sans intervention : attendre le prochain
artefact V1 d'une nouvelle date, vérifier origine automatique de la première
capture, fraîcheur <=2h au démarrage, génération avant chaque match, identités/rangs,
absence de doublons, sauvegarde et restauration. Aucun lancement manuel n'a été
fait pendant cette préparation. Aucun défaut opérationnel nouveau démontré ; pas
de modification spéculative des déclenchements.

## Accord suivant

L'accord demandé porte sur le protocole concret A01 préparation 1, notamment la
maturité à +12h, les exigences de complétude, la revue obligatoire des corrections,
les deux dénominateurs et leurs seuils 98 %. Il autoriserait le gel de la version
retenue, le choix/versionnement d'un T0' futur et le branchement séparé d'A01 avant
ses nouvelles observations. Le préparateur actuel reste exploratoire ; moteur
confirmatoire, prise en compte de corrections approuvées et branchement de production
A01 ne sont pas activés par cette PR. Ni promotion de V2, ni modification des
sélections, ni réutilisation confirmatoire des journées déjà examinées.
