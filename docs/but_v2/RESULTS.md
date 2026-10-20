# Résultats BUT V1 / V2 — replay exploratoire du 20 octobre 2026

**Décision : aucune promotion. Aucun gain global crédible démontré contre BUT V1.**

104284 observations disponibles; 61950 participations communes testées, 1721 matchs, 290 dates et cinq fenêtres. Couverture des lignes test : 100% pour chaque modèle. Les participants sont connus rétrospectivement : cette couverture ne mesure pas les absents pré-match.

| Modèle | Brier ↓ | Log-loss ↓ | Top5 quotidien | Top10 quotidien |
|---|---:|---:|---:|---:|
| v1 | 0.121594 | 0.398709 | 34.07% | 32.14% |
| logistic | 0.122171 | 0.399858 | 34.28% | 33.41% |
| poisson | 0.122352 | 0.402167 | 34.00% | 32.55% |
| hgb | 0.121686 | 0.398602 | 33.52% | 31.90% |

Top5 : 1450 lignes par modèle; Top10 : 2900. Il ne s’agit pas de paris placés. Un meilleur taux Top10 ne démontre pas une meilleure probabilité sur tout l’univers.

| Comparaison à V1 | Δ log-loss | IC95% apparié | Δ Brier | IC95% apparié |
|---|---:|---|---:|---|
| logistic | +0.001149 | [-0.001002; +0.003510] | +0.000577 | [-0.000064; +0.001330] |
| poisson | +0.003458 | [+0.001472; +0.005604] | +0.000757 | [+0.000274; +0.001337] |
| hgb | -0.000107 | [-0.001684; +0.001505] | +0.000092 | [-0.000274; +0.000474] |

Delta négatif = amélioration. La logistique dégrade légèrement les scores globaux sans différence concluante; son Top10 augmente descriptivement de 1,28 point. HGB améliore la log-loss de 0,000107 mais dégrade le Brier; les intervalles englobent zéro. Poisson est moins bon sur les deux scores, y compris après correction des comparaisons multiples. Aucun changement de candidat après lecture de ces résultats.

## Ablations de la logistique

Chaque ligne retire une famille ou conserve une seule fenêtre tirs. Delta positif = retrait défavorable. IC ajusté Bonferroni; les familles ne sont pas causalement indépendantes.

| Variante | Δ log-loss vs complet | IC ajusté | Δ Brier |
|---|---:|---|---:|
| without_volume | +0.000965 | [+0.000479; +0.001438] | +0.000261 |
| without_usage | +0.003639 | [+0.001932; +0.006321] | +0.001015 |
| without_offense | +0.000716 | [-0.000740; +0.001738] | +0.000126 |
| without_team_context | +0.000263 | [+0.000004; +0.000526] | +0.000057 |
| without_calendar | +0.000032 | [-0.000188; +0.000240] | +0.000013 |
| without_drought | +0.000108 | [-0.000559; +0.000649] | -0.000116 |
| without_hot_streak | +0.000027 | [-0.000017; +0.000074] | +0.000013 |
| volume_window_3 | +0.000523 | [+0.000143; +0.000863] | +0.000141 |
| volume_window_5 | +0.000407 | [+0.000075; +0.000701] | +0.000131 |
| volume_window_10 | +0.000003 | [-0.000098; +0.000108] | +0.000005 |

Les tirs et l’usage TOI/PP apportent une information mesurable dans cette logistique. Cela ne rend pas V2 supérieur à V1, qui les utilise déjà. La fenêtre tirs 10 matchs seule ne se distingue pas clairement de 3+5+10; aucun nouveau réglage sélectionné. Le contexte équipe présente un signal faible (Brier ajusté non concluant). Repos, disettes et hot streaks ne montrent pas un apport robuste sur les deux critères après correction. Aucun bonus automatique justifié. Gardien, qualité des tirs/xG et PP1 confirmé : non évaluables.

## Limites et traçabilité

Fenêtres de 60 dates NHL (la dernière 50), refit au début de fenêtre, calibration sur le passé uniquement. Elles peuvent traverser une intersaison. Le V1 comparé suit la même cadence de refit; ce ne sont pas ses probabilités quotidiennes réellement émises. Les anciennes périodes déjà étudiées restent consommées : ce travail est un replay exploratoire, pas un nouveau holdout ni une validation prospective.

Les résultats NHL proviennent du snapshot local déjà présent; ils n’ont pas été recollectés ni indépendamment réhomologués pendant cette mission. L’heure originale de disponibilité des corrections historiques est inconnue. Aucun résultat postérieur au 13 octobre dans le calcul; observations du 17 octobre exclues de la sélection.

Absence de cotes pré-match dans cet historique : ROI non calculable. Les métriques détaillées, intervalles ajustés, calibration par intervalles, precision/recall au seuil .5, segments saison régulière/playoffs, domicile/extérieur et chaque fold figurent dans `results/report.json`. Le seuil .5 sélectionne très peu de joueurs; ces precision/recall ne constituent pas une politique de pari.

Environ 2,44% des features individuelles restent inconnues (début de saison/historique court); conversion inconnue sur 3,60%. Les médianes du train servent seulement au modèle; les données archivées conservent les inconnus.

Les intervalles portent sur les différences de scores, avec blocs circulaires de sept dates NHL et 2000 réplications. Les intervalles extrêmes corrigés sont peu précisément estimés à ce nombre de réplications; les dépendances longues joueur/équipe et la réutilisation de données limitent leur interprétation.

## Reproduction

SHA256 du fichier source exact : `354a4c5385f9b36b14f65410e4caf0585b1b92d25262af5626d92959b0ef083e`.
Les données lourdes et prédictions ligne par ligne restent dans `outputs/but_v2/replay_20261020/` (hors Git). `source_original.csv.gz` conserve les octets du CSV source, vérifiés après décompression. `input.csv.gz` est la copie normalisée; `predictions.csv.gz` contient toutes les probabilités. Rejouer avec le CSV source décompressé dans un nouveau dossier, sans changer la configuration.

Le protocole a été écrit avant fit et ses hashes correspondent aux fichiers exécutés. Son git_sha désigne le commit de départ : le runner était alors non commité, puis figé dans c47a7c1. Les versions Python/dépendances sont indiquées dans les preuves. Pas de second run complet revendiqué; tests de déterminisme exécutés sur fixtures.

Conditions pour poursuivre : accumuler captures shadow pré-match, univers et résultats officiels appariés; conserver les cotes si disponibles; fixer une durée prospective et un calcul de puissance à partir des écarts attendus avant toute décision. Revue humaine obligatoire avant raccordement quotidien et avant une éventuelle promotion.
