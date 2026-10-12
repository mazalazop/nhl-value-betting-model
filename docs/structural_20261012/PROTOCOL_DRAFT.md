# Étude structurelle POINT / BUT — cadrage distinct

Statut : **préparation, pas encore préenregistrée pour exécution**. Aucun fit ni score structurel. Branche `research/structural-20261012`, après clôture de l'étude 201e6ff au commit `a99576f`. Ce document ne modifie pas le protocole de cette étude close.

## Contraintes et sources vérifiées

- POINT profondeur 4 / 81 features reste la référence de production. BUT GOAL17 est la référence de recherche ; C19 sert de comparaison secondaire, sans statut production.
- Holdouts 2024-02-01–2024-04-19 et ancien bloc BUT à partir du 2026-03-29 consommés. Aucun résultat de ces blocs ne peut guider une nouvelle sélection, calibration ou réglage. Pas de réouverture pour expliquer l'échec face au benchmark.
- Données 2022-23 et 2023-24 : 94 429 participations régulières. Le nouveau contexte officiel comprend 5 248 lignes équipe/match, couvrant PP, PK et pénalités sans valeurs manquantes. [Contrats](source_contracts.json), [couverture/hash](team_context_coverage.json). Six rapports saisonniers ont suffi ; cache compressé réutilisable.
- Les rosters saisonniers sont rétrospectifs. **PP1 historique indisponible** : aucune assimilation PP TOI élevé = PP1. Un futur signal PP1 actuel doit avoir sa propre source, date d'observation, preuve d'identité et fraîcheur ; il reste exclu des fits historiques sans archive.
- Aucun pari, aucun changement de sélection ou de publication. Pas de ROI sans cotes historiques fiables.

## Hypothèses à figer avant calcul

H : niveau historique. Exposer séparément N−1 complet, N−2 si disponible, saison courante strictement antérieure : fréquence 1+, production/match, production/60, tirs/60, effectifs/expositions. Ne pas imposer un poids supérieur à N−1 parce que cela paraît logique. Estimer le shrinkage sur le train uniquement ; préciser formule, estimateur et profondeur minimale avant fit. Les saisons manquantes restent inconnues, jamais zéro. Ne pas présenter une saison partielle comme complète.

R : rôle et contexte. TOI et PP TOI réellement observés avant match, part du PP équipe (temps individuels/temps PP équipe, pas somme des joueurs divisée arbitrairement par cinq), couverture, évolution d'usage, absence et changement d'équipe dont l'identité pré-match est établie. PP équipe : `ppOpportunities`, `powerPlayGoalsFor`, `ppTimeOnIcePerGame`. PK adverse : `timesShorthanded`, `ppGoalsAgainst`. Pénalités : `penalties`, `totalPenaltiesDrawn`, taux officiels/60. Les pénalités et PIM ne sont pas identiques au nombre d'opportunités PP.

Estimer les opportunités PP attendues avec des données strictement antérieures, sur le train uniquement. Une valeur prédite doit être désignée comme estimation, jamais comme opportunités réelles futures. Une seule interaction supplémentaire : exposition PP joueur × opportunités attendues. Aucun produit combinatoire automatique.

F5 / F10 : production récente sur cinq ou dix matchs. Les dissocier de l'usage récent dans R. Drought relatif conservé comme hypothèse, sans loi de retour à la moyenne forcée, sans règle « une série doit s'arrêter ».

Comparer séparément pour chaque cible : référence actuelle, H, H+R, H+F5, H+F10, H+R+F5, H+R+F10. Même univers, mêmes dates et même profondeur historique pour toutes les variantes. Le gain lié à l'ajout de saisons n'est pas un gain de feature. Pas de recherche d'architecture/hyperparamètres simultanée.

## Conditions avant préenregistrement exécutable

1. Fixer les inputs hashés et les saisons nécessaires pour N−1/N−2, sans utiliser les blocs consommés pour sélectionner. Les deux saisons supplémentaires seules ne donnent pas deux saisons antérieures complètes à chaque ligne de développement. Compléter le warm-up, ou déclarer les insuffisances explicitement avant définir les fenêtres.
2. Vérifier la couverture du contexte utilisé par la référence POINT, notamment standings. Ne pas comparer un POINT artificiellement privé de contexte à une variante enrichie.
3. Fixer listes exactes, formules, shrinkage train-only, folds, population, seuil de gain utile, correction des comparaisons multiples et méthode bootstrap avant toute métrique structurelle. Le document actuel est un cadrage, pas une autorisation de tuning.
4. Réserver un nouveau bloc **prospectif** avec dates/échéance et règle d'arrêt fixes avant ses premières prédictions. Aucun ancien holdout ne redevient neuf. La taille doit tenir compte des dépendances par date/match ; une petite différence peut demander plusieurs saisons (voir planification précédente).
5. Tester shift, invariance aux résultats courants/futurs, parité et missing vs zéro. Les données équipe de ce dossier sont des résultats bruts : elles ne sont jamais des features du match courant.
6. Mesurer log loss/Brier, IC corrigés, calibration, AP/AUC/top décile, stabilité saison/fold. Promotion seulement après résultat indépendant favorable et référence forte battue ; production inchangée sinon.

## Ordre suivant

Manifest/warm-up et protocole exécutable → tests temporels → POINT structurel → BUT structurel séparé → confirmation prospective. Value/PP1 opérationnel et défenseurs restent conditionnés à la validation probabiliste. Le top10/top5 est un plafond, jamais un quota de paris.

Reprise des sources (les outputs existants ne sont pas écrasés) :

```bash
.venv/bin/python -m pytest tests/test_structural_team_context.py -q -W error
# Rapports déjà collectés : ne pas relancer sans nécessité.
# scripts/collect_structural_team_context.py --history <source.csv.gz> --output <dossier neuf>
```
