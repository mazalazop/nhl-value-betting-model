# Conservation durable — 22 octobre 2026

Le registre Actions (90 jours) est une copie de travail. La copie durable utilise
les releases GitHub **draft**, préfixe `henachel-but-v2-state-`, séparées du registre
POINT/V1. Aucune publication, aucun abonnement ni écriture dans les picks.
Chaque snapshot possède un index complet et des bundles immuables des seuls
fichiers modifiés. Les index référencent les bundles des snapshots précédents :
**conserver toutes ces releases**, pas seulement la dernière.

Rétention prescrite : pendant toute l'étude et au moins 365 jours après sa clôture.
Pas d'expiration programmée ni suppression automatisée. Un administrateur GitHub
peut néanmoins supprimer ces données : ce n'est pas une garantie WORM ni une
copie indépendante du compte GitHub. Une copie hors compte peut être obtenue
avec la commande de restauration ci-dessous, sans nouveau service.

Limites [documentées par GitHub](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases) :
1000 assets par release, chacun <2 GiB, pas de plafond global documenté pour les
releases. Notre budget est 128 MiB cible par bundle non compressé (un fichier plus gros est isolé), au plus 990 bundles
par snapshot; un fichier individuel >512 MiB arrête la sauvegarde et exige une
rotation explicite, sans perte silencieuse. Les sources de training sont déjà
compressées et dédupliquées. Enveloppe de planification sur 180 jours : plusieurs
Go de sources quotidiennes et des rapports cumulatifs croissants; pas de copie
intégrale répétée à chaque run. Les reçus `verified.json` donnent le nombre de
fichiers et les octets réels à surveiller quotidiennement. Aucune limite GitHub
ne doit être interprétée comme une garantie de capacité infinie.

Avant de valider chaque snapshot, téléchargement et restauration **complète**
dans un répertoire temporaire, contrôle SHA256 de tous les bundles/fichiers et
égalité avec `archive_manifest.json`. `index.json` puis `verified.json` sont les
marqueurs de validation. Un snapshot incomplet bloque la reprise; aucun retour
silencieux à une ancienne version. Un artefact Actions plus récent que la copie
durable bloque aussi la capture jusqu'à réparation de sa sauvegarde.

Le job de calcul conserve contents:read/actions:read. Les seuls jobs `restore` et `backup` possèdent contents:write pour accéder aux
assets draft; aucun de ces jobs ne calcule les probabilités.
La concurrence couvre l'ensemble du workflow (calcul ET sauvegarde).

Restauration manuelle, avec GH_TOKEN en environnement et répertoire inexistant :

```sh
python scripts/backup_but_v2_state.py restore --state restored-shadow
# Une vue antérieure référencée par compactions/ :
python scripts/backup_but_v2_state.py restore --state restored-old --snapshot <SHA256>
```

Réparer une sauvegarde interrompue : télécharger l'artefact le plus récent,
relancer `backup --state <répertoire>` avec GITHUB_SHA/GITHUB_RUN_ID d'origine,
vérifier le reçu, puis reprendre. Ne pas effacer de snapshot pour contourner
un contrôle. Le dernier index complet est autonome en métadonnées, mais dépend
des bundles immuables qu'il référence dans les releases précédentes.

Test distant réel préalable : release draft `420617845`, préfixe distinct
`henachel-but-v2-restore-probe-`, deux fichiers techniques, 186 octets; restauration
indépendante effectuée le 22 octobre. SHA256 de l'index :
`2feeed4248c5e281f2dc2855d6e9c5ef8282a4bf84b23e71e9c7b4e41b3ae9fd`.
Ce test n'est ni une prédiction ni T0. Les tests automatisés couvrent aussi les
versions anciennes, la déduplication, les corruptions, les chemins dangereux,
les budgets et les reprises idempotentes.

Capacité de la copie de travail : après restauration intégralement vérifiée,
les anciennes vues `batches/*/{reports,cumulative,outcomes.csv}` sont retirées
**uniquement de la copie de travail**, sauf le dernier bilan. Elles restent
restaurables par leur index de snapshot immuable; `compactions/` enregistre ce
lien et chaque chemin. Captures, dernier historique, sources officielles, entrées,
journaux de capture et dernier bilan restent présents. Cela évite la croissance
quadratique des rapports quotidiens dans le runner. Une restauration manuelle
d'un index ancien permet de retrouver exactement la vue précédente. Les copies
durables ne sont jamais purgées. Le stockage total distant des vues reste
croissant : suivre les reçus, et ne jamais supprimer les bundles référencés.

Le job `source` ignore les fins de runs horaires ne contenant que des annotations
contextuelles (pas de nouvel artefact de prédiction). Un dispatch manuel sans
artefact V1 reste une erreur. Toute copie Actions préexistante sans sauvegarde
durable validée bloque également le démarrage; `initialize` ne contourne pas
ce contrôle.


Mesure de capacité réelle du 22 octobre : historique source du 20 octobre,
249 861 571 octets bruts, 75 027 069 octets gzip niveau 1. Soit environ 13,5 Go
pour 180 versions quotidiennes, hors croissance. Cette mesure sert seulement
au dimensionnement, pas à une observation prospective. Ces 180 versions ne
peuvent pas toutes rester dans le runner standard. La compaction conserve donc
seulement la source training référencée par le dernier batch dans la copie de
travail; les versions antérieures restent **intégralement** dans les snapshots
immuables, avec leur chemin, hash et reçu de compaction. Au redémarrage, un
historique antérieur et le nouvel historique peuvent coexister temporairement.
Les références historiques restent auditables en restaurant le snapshot indiqué
par `compactions/`. Aucun fichier source distant n'est effacé.

Cette correction est exclusivement opérationnelle : aucun fichier du candidat,
du calcul des probabilités, des métriques ou du protocole gelé n'est modifié.


Incident de reprise réel du 22 octobre, run `39772335823` : le jeton
contents:read ne voyait aucune release draft, d'où arrêt protecteur avant tout
calcul. La première capture et sa sauvegarde sont intactes. L'[API GitHub](https://docs.github.com/en/rest/releases/releases)
ne liste les drafts que pour les accès disposant du droit de push. La restauration
est donc isolée dans `restore`, qui possède contents:write mais effectue seulement
les lectures et vérifications du staging. Les entrées passent au job `shadow`
par un artefact temporaire d'un jour; `shadow` conserve contents:read et ne peut
pas publier ni écrire dans le dépôt. `backup` ne démarre que si un registre a
réellement été téléversé. Ce changement de permissions opérationnelles ne modifie
aucun des fichiers du candidat ou protocole gelés à T0.

Découverte résiliente : le run automatique `39801360938` a observé une liste de
drafts vide malgré contents:write; sa relance avec le même code a retrouvé les
copies. La cause interne GitHub n'est pas établie. Si la liste est vide alors
qu'un registre Actions existe, le staging retrouve le reçu du même run main
validé, puis lit **directement** l'ID de release indiqué. SHA256 de l'index,
marqueur de restauration, identité du run et tous les fichiers restent vérifiés.
Le reçu ne remplace pas la sauvegarde durable. Sans reçu valide ou accès direct,
arrêt protecteur, jamais initialisation vide. Après expiration des artefacts,
un opérateur peut utiliser l'ID durable conservé dans le rapport :
`python scripts/backup_but_v2_state.py restore --state restored --release-id <ID>`.

Repli quotidien indépendant : le run V1 `39805484632` lancé par le dispatcher
(bot) a terminé à 16:08:55 UTC sans événement shadow observé à 16:19. La cause
interne n'est pas affirmée; la chaîne ne doit pas dépendre uniquement de la
propagation de `workflow_run`. Le shadow conserve ce déclencheur et ajoute
06:30, 10:30, 12:30 et 16:30 UTC. Le sélecteur recherche parmi les 50 derniers
runs réussis sur main le dernier contenant réellement l'artefact modèle, en
ignorant les mises à jour contextuelles et les runs non fiables. Un dispatch
sans ID utilise exactement cette découverte, sans relancer V1.

Les contrôles temporels du candidat restent inchangés. Un ancien run peut servir
à mettre à jour les résultats d'une capture existante, jamais à fabriquer une
nouvelle prédiction rétroactive. Chaque date conserve sa première paire et son
Top10; aucune nouvelle source contextuelle ne modifie le candidat. Les jobs de
production POINT/V1 et leurs horaires ne sont pas modifiés.
