# HENACHEL : procédure de lancement et validation quotidienne

## Exécution normale
- Workflow : `henachel-points-global` (fichier `.github/workflows/Workflow global Henachel — POINTS.yml`).
- Le job `Pipeline modèle POINTS` doit réellement s'exécuter. Un workflow global vert avec ce job `skipped` **ne constitue pas une production**.
- GOAL expérimental doit produire un fichier daté et exploitable, pas seulement un job d'évaluation BUT V2.
- Vérifier la date NHL de chaque sortie, les jobs et les artefacts, et ne jamais recycler les probabilités d'une journée précédente.
- Vérifier l'absence d'un run complet déjà en cours avant tout déclenchement.

## Si la production du jour n'a pas démarré
1. Privilégier `workflow_dispatch` sur `main` avec `context_only=false`, `goal_research_enabled=true`, et `run_date` du jour NHL. Ne pas déclencher `context_only`.
2. Si le connecteur disponible ne permet pas `workflow_dispatch`, **ne pas conclure que le lancement est impossible** : ce workflow accepte aussi les `push` sur `main` modifiant son propre fichier. Une modification minimale de commentaire de ce YAML déclenche une production POINT et active GOAL expérimental (condition `github.event_name == 'push'`).
3. Ce repli par commit est exceptionnel : il modifie l'historique Git. Ne l'utiliser qu'après vérification des runs du jour et de la branche/configuration, et jamais en boucle.
4. Rechercher le nouveau run et vérifier que `Pipeline modèle POINTS` n'est pas `skipped`.

## Livraison
- Ne produire le classeur que si POINT **et** GOAL ont des sorties du jour fraîches, complètes et cohérentes.
- Référence stricte : `NHL_Henachel_2026-10-20_REMISE_EN_FORME.xlsx`.
- Quatre onglets exactement : `Analyse des matchs`, `Top 10 pointeurs`, `Top 10 buteurs`, `Historique PP1`.
- Probabilités brutes inchangées. PP1 vérifiés. Cotes Unibet horodatées, seuil >= 1,35.
- Vérifier l'existence réelle du fichier avant de donner le lien ; aucune livraison provisoire.
- Si sorties manquantes, rapporter le blocage précisément ; ne pas annoncer une livraison.

## Dette technique
- Mettre à disposition un déclenchement direct `workflow_dispatch` depuis l'outil connecté afin de supprimer le repli par commit.
- Contrôler l'heure de programmation selon Europe/Paris et les changements d'heure (cron GitHub est en UTC).
