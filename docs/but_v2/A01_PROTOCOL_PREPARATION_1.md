# A01 — protocole de préparation 1, non activé

Mise à jour du 24 octobre : définitions et seuils approuvés ; raccordement terminé
dans [A01_STARTUP.md](A01_STARTUP.md). Les limitations de raccordement décrites
ci-dessous décrivent la préparation initiale ; la procédure de démarrage courante
et la revue exécutable des corrections sont celles du document A01_STARTUP.
Aucun démarrage effectif n’est autorisé ni enregistré.

Statut : PR révisable ; **enabled=false, t0_utc=null**. Approbation du principe
reçue le 23 octobre 2026, autorisation d'activation non reçue. Le CLI livré est
exclusivement exploratoire et refuse une configuration qui prétend activer l'étude.
Aucun workflow ne l'appelle. L'étude BUT_V2_PROSPECTIVE_1 continue inchangée.

## Séparation des études et prochain accord

Le prochain accord porterait sur les définitions, règles de maturité et preuves
ci-dessous, puis autoriserait la préparation du démarrage séparé A01 : fixer et
versionner un T0' futur, l'identifiant de registre, les empreintes de la version
retenue et son branchement isolé **avant la première prédiction concernée**.
Le T0' n'est pas la date d'approbation ni une date reconstruite après coup.
Les captures observées avant T0', dont les 22 et 23 octobre, sont exploratoires
pour A01 et ne peuvent entrer dans sa fenêtre confirmatoire.

Ne pas réinitialiser, migrer ou écrire dans le registre original. Les prédictions
futures communes peuvent être référencées par leurs empreintes immuables dans le
registre A01 séparé, sans modifier probabilités, candidat, entraînement ou rangs.
A01 doit conserver son propre historique d'observations/révisions et ses sauvegardes.
La livraison actuelle ne déploie pas ce registre ni de tâche planifiée A01.

## Classification explicite, sans création de défaites

| Catégorie | Critère |
|---|---|
| PARTICIPANT_RESULT_KNOWN | Deux sources officielles finales cohérentes ; ID/équipe concordants, patineur dans l'effectif, TOI >0 dans le boxscore et buts entiers >=0 connus |
| NONPARTICIPATION_CORROBORATED | ID absent du boxscore ET des rosterSpots, uniquement après certification conservatrice de complétude des deux sources finales |
| PARTICIPANT_RESULT_MISSING | Participation établie par TOI >0 et identité concordante, mais buts absents/non exploitables |
| PARTICIPATION_UNKNOWN | Source manquante/invalide, effectif incomplet, TOI inconnu/nul, contradiction ou révision en attente de revue |
| GAME_PENDING_OR_POSTPONED | Avant match, avant maturité sans final, ou match officiellement non final/reporté |

L'absence dans une seule source ne suffit jamais. Un TOI 00:00 ne démontre pas
une non-participation (arrondi possible) : inconnu, pas défaite. Un joueur présent
au roster mais sans statistiques reste inconnu. Si une source obligatoire manque,
le certificat A01 complet n'est pas établi : inconnu même si l'autre source suggère
une participation. Les diagnostics sont volontairement conservateurs.

### Sources et complétude

Endpoints NHL uniquement, avec ID du match exact :
`https://api-web.nhle.com/v1/gamecenter/{id}/boxscore` et `/play-by-play`.
Chaque enveloppe conserve source, observed_at_utc avec fuseau, document brut et
SHA256. Horodatage futur refusé ; final observé avant le début refusé. Deux réponses
séparées de plus de 15 minutes ne sont pas rapprochées comme une preuve synchronisée.

Pour corroborer une absence : les deux documents sont OFF/FINAL et d'accord sur
les deux équipes ; aucun doublon d'identité ; chaque équipe a 18 à 20 patineurs et
1 à 3 gardiens ; les ensembles patineurs/gardiens correspondent exactement entre
boxscore et rosterSpots ; positions reconnues, TOI valide et buts connus pour tous
les patineurs du boxscore ; le play-by-play comporte un événement game-end.
Toute activité du joueur dans les événements contredit une inférence d'absence
des effectifs. Un effectif plus court, même légitime, nécessite revue et
reste inconnu pour l'inférence d'absence. Aucune réparation artificielle ni remplissage.
Il s'agit d'une certification structurelle conservatrice, pas d'une garantie que
l'API NHL ne puisse jamais publier deux réponses cohérentes mais erronées : les
sources partagent le même fournisseur. Les corrections restent auditables.

### Contradictions et corrections

Conflit d'équipe/ID, doublons officiels, final/non-final discordants ou position
contradictoire : inconnu avec motif explicite. Doublon parmi les prédictions :
arrêt du calcul, jamais dédoublonnage silencieux.

Chaque réponse est un nouveau fichier adressé par contenu, sans remplacement.
Le CLI refuse un dossier de sortie existant ; fournir `--previous-evidence` avec
le snapshot du précédent audit pour une reprise. Une observation plus ancienne
que la précédente est bloquée comme retour arrière temporel.

Si une source précédemment finale change la catégorie ou les buts, conserver la
classification précédente, la nouvelle proposée et leurs empreintes ; résultat A01
inconnu, motif OFFICIAL_REVISION_REQUIRES_REVIEW. Le préparateur ne fournit pas de
commande permettant d'approuver automatiquement cette correction. Une future revue
scientifique doit consigner auteur, date, preuves et décision avant une réconciliation
versionnée explicite ; tant qu'elle n'est pas implémentée/revue, le cas reste inconnu.
Un passage normal LIVE → FINAL est un premier règlement, pas une correction.
Même si la catégorie reste identique, les versions brutes différentes sont conservées.

## Dénominateurs et maturité

N = toutes les paires capturées du périmètre original choisi. La couverture
originale S_original/N est **lue dans le rapport original**, avec empreinte de ses
octets, et conservée sans recalcul selon A01. Le CSV de capture est vérifié contre
son manifeste ; les figures originales doivent correspondre à N et à leur quotient.

Pour A01, un match devient évaluable dès réception d'une source officielle finale
valide, ou à début pré-match archivé +12h, le premier des deux. La règle +12h est
une proposition opérationnelle fixée ici, pas une estimation optimisée sur les scores.
Une contradiction de sources finales rend le résultat inconnu, sans sortir la paire
du dénominateur. Un report ne décale pas silencieusement cette échéance.
À la clôture confirmatoire, toutes les paires de la fenêtre entrent dans le dénominateur,
y compris matchs encore reportés/non finaux (mode closure du calcul).

E = paires évaluables ; S = participants réglés ; A = non-participations corroborées ;
R = participants identifiés dont résultat manque ; U = participation inconnue ;
W = matchs évaluables encore en attente/reportés. E = S+A+R+U+W, P = S+R.
Les matchs non encore évaluables sont affichés séparément, pas qualifiés de données
manquantes. À la clôture E=N de l'étude A01.

| Indicateur | Définition |
|---|---|
| Couverture officielle originale | S_original/N, inchangée |
| Part des non-participants | A/E |
| Part des cas réellement inconnus | (R+U)/E |
| Part en attente évaluables | W/E |
| Part totale sans information achevée | (R+U+W)/E |
| Couverture d'information A01 | (S+A)/E, seuil **98 %** |
| Disponibilité parmi participants identifiés | S/P, seuil **98 %** |

Un dénominateur nul donne null, jamais 100 % ; le gate est faux. Les deux gates
sont nécessaires simultanément, jamais suffisants pour une promotion. Conserver
U et W dans le premier gate empêche de faire disparaître des identités difficiles
pour améliorer artificiellement S/P. Ne pas remplacer ces indicateurs par le seul
pourcentage parmi les cas connus.

## Exigences statistiques et sélection conservées

Après T0' : descriptif à J30, une évaluation confirmatoire à J180 sur une fenêtre
fixe de 180 jours ; au moins 90 dates, 20 000 participations appariées réglées et
2 000 positifs. Couverture des journées V1 éligibles >=95 %, audit externe obligatoire.
Brier et log-loss calculés sur les mêmes participants V1/V2 ; gains minimaux utiles
.0003 et .001 ; IC simultanés 97,5 % via bootstrap apparié par blocs de 7 dates,
10 000 réplications, seed 20261020. Calibration ECE par déciles <=.03 et dégradation
contre V1 <=.005 ; robustesse par segment requise, dérive surveillée.

Candidat, variables, configuration, politique de refit et référence V1 identiques
à l'enregistrement original. Top10 figés avant match, aucune substitution d'absents
ou d'inconnus, effectifs réglés/non réglés toujours affichés. Cotes exclusivement
pré-match archivées, rendement fictif à mise unitaire sur les seuls cas admissibles ;
aucune non-participation ne devient LOSS ou VOID sans preuve/règle bookmaker.
Aucun ajustement sur les premières performances, aucune promotion automatique.
Les gains sur des données déjà examinées ne valident pas ce nouveau protocole.

Le préparateur expose les deux gates de qualité, mais **aucune évaluation
confirmatoire ni autorisation de promotion**. Le moteur confirmatoire séparé et
le branchement automatique doivent être revus lors du démarrage autorisé ; les
exigences originales sont conservées dans la configuration et dans cette section,
non remplacées par le seul résultat des deux gates.

## Utilisation préparatoire et validation

```sh
.venv/bin/python scripts/audit_but_v2_participation_a01.py \
  --predictions /copie-registre/captures/AAAA-MM-JJ/predictions.csv \
  --original-report /copie-registre/batches/run/reports/AAAA-MM-JJ/report.json \
  --output outputs/but_v2/a01_audit_unique
```

Collecte officielle actuelle, sans écriture du registre. Pour replay : fournir
`--evidence snapshot.json --as-of HORODATAGE_UTC` et, pour correction,
`--previous-evidence snapshot_precedent.json`. Les résultats restent exploratoires.
Le CLI refuse les chemins à l'intérieur d'un registre original et les captures
altérées. Fichiers : classifications, métriques, snapshot, enveloppes immuables,
index des preuves, erreurs de collecte et provenance du rapport original.

Tests : absence complète, réponses vides/tronquées, résultat manquant, TOI nul/inconnu,
match reporté, contradictions, correction finale, retour arrière temporel, doublons,
absence d'activation, préservation des inputs, refus d'écrasement et empreintes gelées.
