# Étude 201e6ff — close, 12 octobre 2026

**BUT = research only / not production approved.** POINT et le protocole initial sont inchangés. Le holdout historique 2024-02-01–2024-04-19 est consommé définitivement. Aucun retuning après ouverture.

## Données et reproductibilité

- 2022-23 : 1 312 matchs réguliers, 47 208 participations, 951 joueurs.
- 2023-24 : 1 312 matchs réguliers, 47 221 participations, 924 joueurs.
- Total : 94 429 participations, 1 093 joueurs uniques. PP 100 %, aucun champ requis absent. Deux matchs témoins vérifiés exactement contre boxscores ; le reste vient des rapports officiels par match, pas de 2 624 boxscores relus.
- Développement : 75 282 participations, 287 dates, 1 049 joueurs. OOF : 54 265 participations, 208 dates, 998 joueurs, prévalence 15,30 %.
- Reconstruction par fonctions de production : 23 features, 684 lignes sur quatre dates (début de saison et changement de saison inclus), zéro divergence historique/futur.
- Roster saisonnier rétrospectif, pas preuve de disponibilité pré-match. Standings non utilisés par GOAL17 ou les ajouts préenregistrés. Le contexte roster/standings sondé précédemment reste partiel (429 et roster ARI 2023-24 vide), sans imputation à partir de ces sources.

Les sorties volumineuses restent dans `outputs/research_20261005/`. [Hashes](artifact_hashes.json), [couverture](historical_coverage.json), [parité](feature_reconstruction.json), [résultats complets](development_results.json).

## Développement : protocole inchangé

| Variante | Log loss | Brier | AUC | ECE | Décision |
|---|---:|---:|---:|---:|---|
| GOAL17 | 0,404513 | 0,123120 | 0,673331 | 0,003644 | référence |
| A — volume/variabilité | 0,403193 | 0,122830 | 0,675272 | 0,005280 | rejetée : fold 4 dégradé |
| B — interactions | 0,404017 | 0,123020 | 0,674905 | 0,003987 | rejetée : instable, IC inclut zéro |
| C — calendrier/transfert | 0,402890 | 0,122846 | 0,675571 | 0,002791 | retenue en développement |

C améliore log loss/Brier dans les quatre folds. Δlog loss = −0,001623 ; IC bootstrap par blocs de sept dates, corrigé pour trois familles : [−0,003150 ; −0,000531]. Aucun assemblage de familles. Baseline AP 0,26139, top décile 32,36 % ; les métriques détaillées et calibrateurs par fold sont archivés. Ce nouvel historique n'est pas directement comparable aux scores de l'étude précédente.

C a été verrouillée et poussée dans `4ef1ec8` avant ouverture : 19 features (GOAL17 + `team_games_previous4days_pre`, `player_team_changed_pre`), HGB profondeur 4, feuille 50, seed 42, mêmes paramètres et politique de calibration. Baseline et C finales : sigmoid sélectionnée et apprise uniquement avant le 1er février 2024. Voir [verrou](candidate_locked.json).

## Confirmation historique, une seule ouverture

19 147 observations, 532 matchs, 821 joueurs, 74 dates. Prévalence 15,07 %.

| Modèle | Log loss | Brier | AUC | AP | Top décile | ECE |
|---|---:|---:|---:|---:|---:|---:|
| GOAL17 | 0,392483 | 0,119712 | 0,699664 | 0,278270 | 33,84 % | 0,010061 |
| C verrouillée | 0,392025 | 0,119611 | 0,700832 | 0,280203 | 33,73 % | 0,010287 |
| Joueur lissé | 0,392754 | 0,119479 | 0,697023 | 0,284937 | 34,67 % | 0,007217 |

- C vs GOAL17 : Δlog loss −0,000458, IC95 % [−0,000794 ; −0,000155] ; ΔBrier −0,000102. Gain probabiliste modeste confirmé contre GOAL17.
- C vs joueur lissé : Δlog loss −0,000729, IC95 % [−0,002401 ; +0,000909] ; ΔBrier **+0,000132**. Avantage non démontré, Brier et top décile moins bons.
- **Pas de promotion en production.** C reste un candidat de recherche amélioré par rapport à GOAL17, pas un vainqueur confirmé contre le benchmark difficile. Aucun changement des règles après ce résultat.

[Résultat final](final_result.json) et [marqueur de consommation](HOLDOUT_CONSUMED.json) versionnés. La commande de clôture refuse une autre ouverture, même dans un nouveau dossier. Le protocole initial n'a pas été modifié.

## Limites et suite

Deux saisons supplémentaires seulement ; dépendance entre joueurs/matchs et bootstrap par dates imparfait. Gain C faible. Les prédictions sont conditionnelles aux joueurs effectivement observés ; un transfert dérivé de l'équipe du match suppose qu'une future source roster fournisse cette identité avant match. Les rosters saisonniers ne prouvent pas cette disponibilité historique. Le holdout est rétrospectif, pas une campagne prospective, et les hypothèses avaient été informées par des études d'autres saisons. Aucune preuve de rentabilité ni cotes historiques fiables.

La famille C combine calendrier et changement d'équipe : cette étude ne sépare pas leur contribution. Aucun essai supplémentaire n'est lancé pour l'expliquer sur le holdout consommé. Toute étude structurelle suivante doit avoir un protocole distinct et une nouvelle confirmation prospective.

Commandes de contrôle :

```bash
.venv/bin/python -m pytest tests/test_research_history.py tests/test_research_context.py tests/test_research_goal_temporal.py tests/test_research_goal_holdout.py tests/test_goal_model.py tests/test_goal_pk.py -q -W error
git diff 201e6ff -- docs/research_20261005/protocol.json
git diff --check
```

Le développement est reproductible avec `scripts/research_goal_temporal.py` et les inputs hashés ; ne pas relancer les expériences sans nécessité. Aucun modèle POINT/BUT de production ni code partagé modifié.
