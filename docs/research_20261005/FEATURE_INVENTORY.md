# Inventaire consolidé, 5 octobre 2026

[feature_inventory.csv](feature_inventory.csv) contient **81 entrées POINT et 17 entrées BUT** (une ligne par cible/feature), consolidées à partir des preuves déjà versionnées. Aucun entraînement, nouvelle permutation ou réouverture du holdout n'a été effectué.

- **POINT** : baseline de production profondeur 4, features inchangées. Les importances par permutation et ablations reprises ici ont été mesurées avec la **baseline profondeur 6 de l'étude du 3 octobre**, et non avec le modèle depth4 actuel. Elles sont diagnostiques, pas causales. Une permutation déterministe par feature/fold, au maximum 2 000 lignes espacées : pas une estimation exhaustive de l'importance.
- **BUT** : candidat de recherche à 17 features, non approuvé pour production. Aucune importance individuelle ni ablation individuelle BUT n'a été publiée. Les champs correspondants sont `unavailable`; les importances POINT ne sont jamais transférées à la cible BUT.
- Les statistiques descriptives existantes portent sur le développement **avant le 29 mars 2026**, saisons 2024-25/2025-26. Pour les 12 colonnes partagées par BUT, mêmes définitions et même snapshot : statistiques reprises avec cette provenance explicite. Les corrélations reprises sont les trois corrélations avec l'univers des features **POINT**, pas une nouvelle matrice BUT. Pour les cinq colonnes propres à BUT, données manquantes/stabilité/corrélations individuelles non publiées restent `unavailable`.
- NaN après construction ne signifie pas indisponibilité brute. Les defaults fixes masquent parfois une profondeur insuffisante : fréquences BUT inconnues remplacées par 0,20, moyennes buts inconnues par 0. Le CSV décrit les comportements actuels sans les modifier. PP inconnu ne doit pas être assimilé à un zéro observé.

## Classement justifié

**Utiles au niveau du groupe** : streaks POINT (leur retrait dégrade les quatre fenêtres), drought BUT à deux features (amélioration des quatre fenêtres de développement, sans promotion finale). Cela ne prouve ni l'utilité de chaque variable du groupe séparément, ni le caractère indispensable d'une variable.

**Redondance diagnostique** : `redundant_diagnostic_only` signale une corrélation absolue >=0,95 dans les trois corrélations rapportées. Ce n'est pas une autorisation de suppression : des variables corrélées peuvent se remplacer et les suppressions étudiées ne passent pas le seuil de promotion.

**Faibles / instables / suppression éventuelle** : importance faible ou négative et ablation instable sont consignées, mais aucune feature n'est démontrée nuisible ou équivalente à son absence. Aucune suppression retenue. Les autres groupes restent `insufficient_or_unstable_group_evidence`; aucune étiquette «essentielle» n'est inventée.

## Expériences déjà conclues : ne pas répéter à données/protocole identiques

- POINT : 11 suppressions de groupes (head-to-head, scoring récent, retour/absence, profondeur historique, calendrier, saison, tirs, standings, streaks, forme équipe, usage). Aucun retrait retenu.
- POINT : drought relatif, drought surprise, part PP, tendance tirs 5–10, tendance TOI 5–10. Aucun ajout retenu. Seul depth4 a été promu; feuille100 non retenue.
- BUT : conversion/tendance tirs, usage TOI/part PP, contexte équipe/adversaire, retour, drought, PK. Seul drought retenu en développement. Depth6 et feuille100 non retenus. Le candidat final ne démontre pas sa supériorité sur le benchmark joueur lissé : holdout consommé définitivement.

Une nouvelle comparaison de ces familles exige un nouvel élément préenregistré (historique supplémentaire réellement validé, définition différente motivée, nouveau développement). Les anciennes dates finales ne peuvent pas servir à sélectionner cette définition. Les données de lignes, PP1/PP2, blessures du jour et gardien probable demeurent exclues sans archive pré-match horodatée.
