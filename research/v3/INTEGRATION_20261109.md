# Vérification de compatibilité du 9 novembre 2026

La branche de recherche a été synchronisée avec `main` au commit `c49aff9d098abdd0e95a3dc4f0fad60dde39e377`, par fusion de main dans la branche de recherche. La PR #21 reste ouverte ; aucune fusion vers main ni activation V3 n'a été effectuée.

Depuis la livraison du 2 novembre, main a reçu des évolutions indépendantes de collecte, des backfills et des actualisations postgame. Les modèles, configurations et documents BUT V2 n'ont pas changé entre la base scientifique `3f90a4a` et cette base d'intégration. Le diff de la PR reste limité à `research/v3/` et `tests/test_v3_research.py`.

Les résultats de recherche, leurs entrées et leurs empreintes restent ceux figés au 1er novembre. Aucun nouveau résultat, backfill ou correction ultérieure n'a été ajouté aux expériences. Le bilan CANONICAL de 174 picks n'est donc pas le cumul courant du 9 novembre. Les chiffres restent vérifiables dans leur contexte de capture, sans prétendre être actualisés.

`protected_files.json` conserve les 467 empreintes de la base scientifique originale. `integration_baseline.json` ajoute les 858 empreintes opérationnelles et historiques de main au 9 novembre. `verify_delivery.py` vérifie les fichiers de recherche et l'égalité avec cette base d'intégration ; les anciens manifestes ne sont pas écrasés.

Le test V3 de préservation vérifie désormais les preuves numériques figées et la désactivation des quatre configurations. Le contrôle des fichiers opérationnels courants reste dans la vérification de livraison. Une référence figée aux vues postgame vivantes ne doit pas empêcher les actualisations quotidiennes ou les futurs changements indépendants autorisés de main. Les tests existants de POINT, BUT, A01, Shadow et sauvegardes restent inchangés.

La CI du 2 novembre (`41413956656`, 459 tests) ne suffit pas à certifier cette nouvelle intégration. La nouvelle CI de la PR valide la révision synchronisée ; son lien et son résultat sont indiqués dans la description de la PR.
