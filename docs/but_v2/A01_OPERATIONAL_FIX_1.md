# A01 — correction opérationnelle 1 (1 novembre 2026)

## Autorisation et périmètre

Le propriétaire a explicitement autorisé l'investigation, la correction, les tests,
la PR puis la fusion après contrôles de sécurité réussis (« MISSION ASTRA : Audit,
correction et sécurisation des workflows HENACHEL BUT V2 »). Cette autorisation
ne permet aucun changement de modèle, de sélection, de protocole statistique,
de T0, d'exclusion ou de résultat confirmatoire figé.

La référence `A01_RELEASE_1` et `configs/but_v2_a01_start.json` restent identiques.
`configs/but_v2_a01_operational_patch.json` relie les empreintes originales et
corrigées de quatre fichiers d'exécution exclusivement. Le validateur refuse un
changement d'un autre fichier de la référence, une mauvaise référence ou une
empreinte différente. Chaque nouveau batch prospectif archive ce manifeste ;
le registre conserve les versions précédentes. Il ne s'agit ni d'un nouveau
candidat ni d'une réinscription de l'étude. Les contrôles de date de préinscription,
les seuils, les dénominateurs, la fenêtre confirmatoire et la promotion humaine
restent inchangés.

## Diagnostic confirmé

L'API GitHub donne **14 échecs sur 19 exécutions les 30–31 octobre**.
Le [run 41248771935](https://github.com/mazalazop/nhl-value-betting-model/actions/runs/41248771935)
échoue dans `cycle()` lors de l'écriture du dossier
`f19d084febe74d99960b5fbdb0ecca163f63554ddf764f8071665f9fe3c61240`.
Son archive contient le joueur 8478864, match 2026020210, précédemment certifié
participant avec 0 but. La nouvelle preuve vaut exactement `{}` : SHA-256
`44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`.

Deux causes se cumulent :

1. `audit()` exigeait seulement que l'ancienne réponse soit finale. Une collecte
   devenue vide/partielle ouvrait donc abusivement une correction officielle.
2. Le nom du fichier utilisait le hash de la preuve courante, tandis que son
   contenu incluait `as_of_utc`. Les évaluations de 20:17:50 puis 22:36:57 UTC
   réutilisaient la même identité pour deux contenus différents.

La collision est reproduite localement avec les données archivées et le code
`main` antérieur à la correction. Les 14 batches interrompus sont conservés.
Les erreurs réseau étaient écrites **après** `cycle()` : lors d'un échec, leurs
motifs précis étaient perdus. L'archive prouve le manque de données ; elle ne
permet pas d'attribuer chaque réponse vide à un code HTTP particulier.

## Correction et traçabilité

- Une révision exige deux sources officielles finales complètes et cohérentes,
  de part et d'autre, et une contradiction réelle entre résultats certifiés.
- La dernière tentative reste distincte du dernier résultat confirmé. Un échec
  de collecte n'annule pas un résultat certifié ; le rapport expose la tentative
  incertaine et le nombre de résultats conservés. Sans preuve antérieure, le
  résultat reste inconnu/en attente, jamais perdant par défaut.
- Les vrais dossiers non résolus restent en attente malgré une panne ou une
  réponse ultérieure non contradictoire. Une décision autorisée est toujours
  nécessaire. Les faux dossiers historiques restent archivés ; ils ne sont pas
  présentés comme contradictions officielles actives.
- Les preuves de dossier version 2 sont adressées par le contenu canonique
  complet, hors horloge d'évaluation. La source et l'heure de collecte restent
  dans l'enveloppe brute adressée par hash. Deux collectes identiques à des
  heures différentes conservent donc deux enveloppes, mais le même dossier
  sémantique. Un rejeu du même contenu ne réécrit aucun fichier.
- `write()`, les contrôles d'intégrité, les décisions immuables, les registres
  distincts et les sauvegardes append-only restent actifs. Aucun ancien fichier
  de preuve, capture, rapport ou dossier n'est supprimé ni migré destructivement.
- Les tentatives et leurs erreurs sont archivées et scellées **avant** le calcul.
  Les réponses HTTP temporaires, timeouts et JSON invalides disposent de trois
  essais au maximum, backoff exponentiel, `Retry-After` plafonné à 30 secondes.
  Un budget global de collecte de 10 minutes évite qu'une panne générale empêche
  la sauvegarde. Le cache final fiable n'est plus remplacé par une tentative vide.

## Politique de déclenchement

La chaîne V1 → Shadow demeure intacte. L'événement A01 `workflow_run` compare les
empreintes des captures admissibles dans les deux index de sauvegarde. Sans
nouvelle capture, il produit `trigger-decision.json` puis évite restauration
volumineuse, évaluation et sauvegarde redondantes. Une capture modifiée atteint
les contrôles d'intégrité et est rejetée ; elle n'est pas silencieusement ignorée.

Les quatre contrôles périodiques **07:45, 11:45, 13:45 et 17:45 UTC** restent actifs,
indépendamment d'une nouvelle capture. Ils récupèrent les résultats tardifs et
recherchent les corrections durant les sept premiers jours. La relecture des
finales plus anciennes reste disponible avec `refresh_official=true`, conformément
au comportement existant. GitHub peut retarder les crons ; aucune ponctualité
absolue n'est promise. Une évaluation manuelle demeure possible avec
`exploratory=false`, `initialize=false`. La concurrence globale reste sérialisée,
`cancel-in-progress: false`.

Un run filtré n'est **pas** une preuve de cycle complet. Les preuves de déploiement
doivent montrer `restore`, `evaluate`, `backup` réellement exécutés et les reçus
de restauration distante. Ne jamais utiliser `initialize=true` pour récupérer
ce registre existant.

## Validation et retour arrière

Les fixtures hors réseau couvrent les réponses identiques, les horloges différentes,
les indisponibilités, les réponses vides/partielles, un seul endpoint, les erreurs
HTTP/JSON, les limites de retries, les vrais buts corrigés, la revue humaine,
l'absence corroborée, TOI nul/inconnu, les matchs non terminés/reportés, les doublons,
la réutilisation interdite d'un run, les déclenchements rapprochés, l'intégrité
scientifique et la restauration après échec intermédiaire.

Avant déploiement, le snapshot A01 `4d0ec4fb9a45e655ec71919e8341c7013ffd949d3866d4705d4328cc481cdf32`
(release 430478731) a été entièrement restauré : **1 454 fichiers, 216 913 878 octets**.
Les journaux et inventaires de validation sont conservés localement dans
`outputs/but_v2/repair_20261101/`. Le rapport de déploiement donne les contrôles CI,
les runs réels et la comparaison des inventaires après correction.

En cas de régression, désactiver temporairement **la variable A01 uniquement**
(`HENACHEL_BUT_V2_A01_ENABLED=false`), conserver toutes les sauvegardes et restaurer
un reçu vérifié dans un nouveau répertoire pour diagnostic. Ne pas modifier
`start.json`, ne pas supprimer les nouveaux dossiers et ne pas remettre en service
l'ancien code connu pour ses collisions. Un correctif ultérieur nécessite une
nouvelle version opérationnelle révisée ; aucune réécriture du protocole original.

Les rapports anciens restent consultables et les nouveaux rapports identifient
l'erratum. La correction peut rétablir dans les rapports suivants des résultats
certifiés que les pannes rendaient artificiellement inconnus ; ce n'est pas une
nouvelle observation sportive, ni une amélioration prédictive démontrée.
