# A01 RELEASE 1 — raccordement prêt, démarrage non autorisé

Les définitions, la maturité final officiel ou début archivé +12h, et les deux
seuils 98 % ont été approuvés. Ce raccordement termine la préparation technique.
`configs/but_v2_a01_start.json` reste disabled, sans identifiant ni T0 effectifs.
La variable HENACHEL_BUT_V2_A01_ENABLED n'est pas activée. L'étude originale,
son registre, ses fichiers gelés et ses sélections restent indépendants.

## Architecture intégrée

Le workflow BUT V2 A01 isolated evaluation propose trois jobs : restore (accès
aux drafts), evaluate (lecture seule GitHub), backup (écriture draft isolée).
Sérialisation globale but-v2-a01-registry, cancel-in-progress=false, et verrou local
pour le CLI. Ni réentraînement ni sélection ne sont exécutés par ce raccordement.

Restore télécharge et vérifie le registre original et celui d'A01 ; evaluate copie
seulement les captures admissibles, conserve leurs probabilités/rangs, collecte
les deux endpoints officiels et écrit une nouvelle révision de rapport ; backup
publie des bundles immuables dans des releases draft et restaure intégralement
chaque sauvegarde avant son marqueur verified. Les erreurs de collecte sont archivées.
Les registres sont interdits dans des chemins imbriqués l'un dans l'autre.

Un lancement manuel **exploratory=true** est un test explicite, pas une activation :
registre A01_EXPLORATORY, namespace durable `henachel-but-v2-a01-exploratory-`.
Pour la toute première exécution, fournir initialize=true ; les reprises utilisent
initialize=false. Si une sauvegarde manque mais qu'un artefact précédent existe,
l'initialisation est refusée : restaurer le reçu durable, jamais recréer à vide.
Aucune observation n'y est confirmatoire. La production future utilise un namespace
`henachel-but-v2-a01-{study_id}-` distinct ; aucun changement de mode d'un registre
existant n'est accepté. Un run ID déjà traité ne peut pas être réutilisé avec des
inputs différents. Une reprise nouvelle réutilise les captures sans doublons.

Conserver tous les bundles et index au moins jusqu'à 365 jours après la clôture,
sans purge automatique. Actions (90 jours) reste une copie temporaire seulement.
Les drafts ne sont pas WORM : une suppression par administrateur reste possible.
Pas de service payant ajouté. Tester régulièrement une restauration dans un chemin
neuf avec le namespace ou l'ID de release du reçu ; comparer l'inventaire complet.

## Revue explicite des corrections officielles

Le registre contient `cases/{case_id}/{evidence_sha}.json`, les preuves intégrales,
les classifications précédente et proposée. Les preuves sont adressées par contenu.
Une correction non revue demeure inconnue à chaque reprise : la référence acceptée
n'avance pas silencieusement. Les rapports précédents sont immuables.

1. Examiner les deux sources officielles, leurs horaires, identités et révisions.
2. Préparer une PR ajoutant un objet à `configs/but_v2_a01_reviews.json` :
   `case_id`, `choice` (ACCEPT_CURRENT ou KEEP_PREVIOUS), `reviewer`,
   `reviewed_at_utc`, `rationale`, `review_reference` (lien de revue humaine).
3. Faire approuver puis intégrer cette décision humaine. Ne jamais synthétiser
   une approbation avec une simple réussite de CI. L'accès main reste la frontière
   de confiance ; le moteur ne prétend pas fournir une signature cryptographique.
4. Au cycle suivant, l'identité du cas lie les documents précédents et nouveaux ;
   une décision hors cas, datée avant ses preuves, future ou réécrite est rejetée.
   La décision, la révision et les résultats sont conservés séparément. KEEP_PREVIOUS
   conserve la référence déjà acceptée ; ACCEPT_CURRENT avance cette référence.
5. Une nouvelle correction officielle crée un autre cas. Une décision existante
   n'est jamais écrasée. Toute rétractation d'une décision exige une évolution
   explicitement revue ; aucun raccourci n'est fourni par le raccordement.

## Rapports et fenêtre confirmatoire

Chaque batch conserve les catégories, les dénominateurs, règlement A01,
comparaison appariée Brier/log-loss/calibration, Top10 figés et ROI fictif uniquement
sur cotes archivées. Les absents ne deviennent jamais LOSS. Le rapport original
**global** et la couverture officielle du sous-ensemble importé sont affichés
séparément. Rapports par date et équipe ; les segments domicile/extérieur et type
de compétition nécessitent un rapprochement officiel séparé avant la revue de
robustesse et restent une obligation humaine, pas un gate implicitement validé.

En mode prospectif, la fenêtre fixe commence à first_nhl_date et dure 180 dates
calendaires. Avant sa clôture, pas d'évaluation confirmatoire. À la clôture toutes
ses paires entrent dans le dénominateur, même reportées. Le calcul vérifie minima
90 dates/20000 participations/2000 positifs, deux seuils 98 %, gains minimaux Brier
.0003/log-loss .001, IC simultanés 97,5 % (10000 réplications par blocs de sept dates),
et calibration ECE <=.03, dégradation <=.005. Couverture des journées éligibles
>=95 % et robustesse segmentaire restent des revues externes obligatoires.
Aucune combinaison de gates ne produit une autorisation de promotion.
Les évolutions de données après clôture produisent une révision descriptive,
pas une nouvelle occasion de sélectionner une meilleure fenêtre.

## Procédure exacte, seulement après accord final

1. Choisir un identifiant d'étude neuf. Revoir la version de PR intégrée et les
   empreintes de `configs/but_v2_a01_release.json` ; toute modification ultérieure
   des fichiers qu'il couvre doit avoir une nouvelle version, jamais passer inaperçue.
2. Faire l'inventaire des dates/captures déjà examinées. Fixer
   excluded_through_nhl_date à leur dernière date ; first_nhl_date doit être
   strictement postérieure, y compris si des captures futures ont déjà été examinées.
3. Préparer la préinscription dans start.json : study_id, approved_by,
   approval_reference, registered_at_utc, t0_utc **futur**, first_nhl_date,
   excluded_through_nhl_date, release_sha256 (digest JSON canonique du fichier release).
   Faire intégrer cet enregistrement sur main **avant T0 et avant les prédictions**. Le stage vérifie également que le dernier
   commit GitHub touchant cet enregistrement précède T0 ; un fichier antidaté
   ajouté après T0 est refusé.
   Aucun T0 effectif n'est fixé par la présente livraison.
4. Après vérification du commit de préinscription, activer enabled dans le même
   enregistrement approuvé et la variable HENACHEL_BUT_V2_A01_ENABLED=true.
   Le moteur refuse l'exécution avant T0. Chaque capture admissible doit avoir été
   générée après T0, appartenir à une date nouvelle admissible et précéder les matchs.
   Les captures historiques restent explicitement exclues et consignées dans imports.json.
5. Lancer le premier cycle avec initialize=true dans le nouveau namespace ;
   les cycles suivants utilisent initialize=false. Au premier cycle, vérifier registration.json, release.json, namespace distinct,
   source originale vérifiée, identités/rangs/horodatages, rapports et premier reçu de
   sauvegarde restaurée. Vérifier ensuite un vrai événement automatique et la reprise.
6. Désactivation : variable false ; préserver intégralement le registre et ses
   sauvegardes. Ne jamais réinitialiser pour remplacer une étude ni changer T0.

L'accord final autoriserait ces opérations de démarrage, pas une promotion de V2,
ni un réglage du candidat, ni la réutilisation confirmatoire des jours déjà examinés.
