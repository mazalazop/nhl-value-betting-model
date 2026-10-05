# BUT : prochaine validation indépendante

`BUT = research only / not production approved`.

Le candidat à 17 features reste figé. Le holdout 2026-03-29–2026-10-02 est consommé : aucun nouveau réglage, choix de feature, choix de calibrateur ou seuil de promotion ne doit être évalué sur ce bloc. Son résultat non concluant reste conservé.

## Voie recommandée

Deux chantiers complémentaires : élargir le développement avec les saisons antérieures, puis confirmer sur des matchs réellement futurs. Les anciennes saisons ne remplacent pas une confirmation prospective du modèle déjà sélectionné sur 2024–26.

1. **Collecter 2022-23 et 2023-24 dans un espace isolé.** Six sondes officielles ont répondu HTTP200 : boxscores finaux, rapport PP joueur et standings historiques à 32 équipes. Les champs nécessaires sont présents sur ces exemples. Voir [les observations](goal_science_20261004/older_season_availability.json). Ce n'est pas encore une preuve de couverture complète ou de compatibilité bout en bout. Le `total=10000` du rapport PP ne doit pas être interprété comme le nombre exact de participations d'une saison; employer les fenêtres de dates/pagination contrôlées du collecteur existant.
2. **Valider avant entraînement.** Inventorier le calendrier régulier/playoffs, la couverture de chaque match/joueur, les identités et alias historiques (notamment Arizona/Utah), TOI/PP absent vs zéro, dates/API des standings, duplications et corrections de source. Refaire seulement les contrôles pertinents et la parité des nouvelles saisons. Aucune donnée réelle existante ne doit être remplacée.
3. **Employer les saisons supplémentaires pour le développement chronologique.** Tout fit précède sa calibration et son évaluation. Une évaluation de 2023 ne peut pas être présentée comme une prévision historique issue d'un modèle entraîné en 2025. Les fenêtres 2024–26 déjà consultées restent des données de développement/diagnostic connues, pas une nouvelle réserve indépendante. L'ancien holdout reste exclu des choix et de toute recalibration expérimentale destinée à le rendre meilleur.
4. **Préenregistrer la confirmation future avant les premiers matchs inclus.** Garder d'abord le candidat actuel et le benchmark lissé de poids 10. Définir le début, la fin ou la taille d'échantillon, la population, les exclusions techniques, les métriques, le seuil minimal utile et la règle de succès. Les mêmes joueurs/matchs sont évalués pour modèle et benchmark. Conserver régulière et playoffs séparés.
5. **Archiver des prédictions de recherche avant match.** Proba, IDs NHL, horaire de début, heure de prédiction, liste des features, couverture/roster, hash des entrées, code et calibrateur. Aucun pari, aucune ingestion de cotes BUT ni publication commerciale. Aucun résultat courant ne peut être une feature. Les observations ultérieures peuvent alimenter les rolling features pré-match, mais les paramètres du modèle et du calibrateur restent gelés pendant chaque bloc final. Ne pas utiliser les labels de ce bloc pour réentraîner/calibrer en cours de confirmation.
6. **Une analyse finale à l'échéance prévue.** Pas de consultation répétée des métriques pour décider d'arrêter quand elles deviennent favorables. Contrôles techniques et de couverture possibles sans consulter l'avantage prédictif. Comparaison principale appariée de log loss contre la fréquence joueur lissée; Brier et calibration comme conditions complémentaires, AP/AUC/top10 comme diagnostics. Bootstrap par blocs de dates, avec limites de dépendance entre joueurs/équipes documentées. Un résultat neutre reste neutre.

Si de nouvelles features sont étudiées sur les données de développement supplémentaires, elles doivent être verrouillées avant le début de leur propre bloc futur. Ne pas essayer rétroactivement plusieurs candidats sur un bloc prospectif déjà commencé pour choisir le gagnant. Les lignes, PP1/PP2, gardiens probables et xG restent exclus sans source historique horodatée vérifiable.

## Ordre de grandeur de l'échantillon

Calcul de planification uniquement, sans entraînement ni réutilisation des labels du holdout. Source : 46 866 observations régulières de développement sur 167 dates NHL, probabilités OOF du candidat et benchmark joueur lissé. L'écart-type bootstrap apparié de la différence moyenne de log loss est environ 0,001456 (blocs de 7 dates, 10 000 réplications).

Approximation : `dates_cibles = dates_source × ((1,96 + 0,842) × SE_source / gain_minimal)^2`, alpha bilatéral 5%, puissance 80%. Résultats dans [le calcul sauvegardé](goal_science_20261004/sample_size_planning.json).

| Gain minimal de log loss à détecter | Dates NHL estimées | Participations joueur/match estimées |
|---|---:|---:|
| 0,001 | 2 779 | 779 884 |
| 0,002 | 695 | 195 042 |
| 0,003 | 309 | 86 717 |

Ce ne sont ni des garanties ni des seuils de promotion supplémentaires décidés après le holdout. L'extrapolation suppose une variance, une densité de matchs et une dépendance temporelle stables. Le changement de saison et le faible nombre de folds peuvent invalider ces hypothèses; une analyse par blocs plus longs et une collecte sans trous seront nécessaires. Les lignes d'un même match ne sont pas des observations indépendantes.

La saison 2026-27 peut fournir un premier bloc prospectif utile, mais **il serait injustifié de garantir qu'une saison suffit pour prouver un très petit gain**. Pour un objectif de l'ordre de 0,003, prévoir potentiellement une validation sur plusieurs saisons et fixer l'échéance avant lecture des résultats. Le gain minimal doit avoir un sens pour la qualité des probabilités, pas être choisi pour obtenir un test significatif. Le ROI reste non mesurable sans historiques de cotes horodatées fiables.

## Défenseurs buteurs : plan conditionnel

Aucun modèle ou marché défenseurs de production n'est commencé.

1. Attendre l'acceptation scientifique de BUT global sur une validation indépendante.
2. Définir le sous-univers défenseurs avec la position NHL disponible avant le match; gérer changements de position/roster sans information future.
3. Mesurer d'abord le modèle BUT global sur ce sous-groupe, avec calibration, Brier/log loss, AP, prévalence et tailles d'échantillon. Comparer à une fréquence joueur lissée propre au sous-groupe.
4. Décider sur le développement seulement si une recalibration dédiée ou un modèle distinct est nécessaire. Les buts étant plus rares, refaire le dimensionnement; ne pas recycler le nombre de lignes global comme preuve de puissance suffisante.
5. Verrouiller la variante, puis confirmer sur un bloc futur indépendant. Traiter les analyses de sous-groupes et variantes comme comparaisons multiples.
6. Après confirmation, intégrer les marchés/IDs, edge et classement en sandbox. Le top5 est un plafond, jamais une obligation de produire cinq paris.

## Jalons produit

- POINT : scientifique, pipeline et Sheets validés; conserver le comportement figé.
- BUT : recherche reproductible et candidat disponibles; avantage sur benchmark difficile non confirmé. Pas d'Unibet BUT ni de sélection en production.
- Suite : disponibilité complète des saisons antérieures → développement temporel contrôlé → protocole futur gelé → confirmation → intégration BUT sandbox → contrôles matching/persistance/settlement/publication → workflow réel validé.
- Défenseurs : seulement après le jalon scientifique BUT, puis validation spécifique et intégration séparée.

Estimation de pilotage du produit final POINT + BUT + défenseurs : **environ 60%**, non mesure statistique. Cette estimation valorise le socle commun et POINT opérationnel, mais considère encore ouvertes la confirmation BUT, son intégration et toute la validation défenseurs. Elle ne constitue pas une promesse de délai ou de rentabilité.
