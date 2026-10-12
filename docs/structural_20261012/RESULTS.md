# Étude structurelle close — aucune substitution admissible

Préenregistrement `4554df1`, préparation/parité `b49037e`. **52 fits HGB effectués, sans tuning ni fit supplémentaire** : 24 POINT, 28 GOAL. `main` conserve POINT depth4 ; BUT reste **research only / not production approved**. Aucun candidat structurel verrouillé, aucune promotion.

## Données et validation

94 429 participations, 2 624 matchs, 1 093 joueurs ; 2022-23 et 2023-24 régulières. Contexte special teams : 5 248 lignes. Standings J−1 : 361 dates demandées, 11 488 lignes, deux dates officiellement vides. Couverture des jointures : 99,9237 % / 99,7713 %.

Parité historique/futur : **138 features, 360 lignes, quatre dates, zéro divergence**. Dates effectives : 2022-10-07, 2023-10-10, 2023-11-01 et 2024-02-05 (premier match disponible après le 1er février). Missingness par saison et hashes dans `preparation.json`. Aucun téléchargement répété.

OOF commun : **42 186 observations, 161 dates, 915 joueurs** ; quatre fenêtres de développement fixées dans `protocol.json`. Tous les anciens blocs observés restent du développement, pas une confirmation indépendante. Moyennes ci-dessous pondérées par observation ; la sélection exige également quatre folds favorables et IC corrigés.

| Cible | Variante | Log loss | Brier | ECE |
|---|---|---:|---:|---:|
| POINT | V0 production | 0.598018 | 0.205092 | 0.006304 |
| POINT | V1 H | 0.599565 | 0.205704 | 0.012185 |
| POINT | V2 H+R | 0.597171 | 0.204669 | 0.013314 |
| POINT | V3 H+F | 0.598991 | 0.205506 | 0.008409 |
| POINT | V4 H+R+F | 0.597630 | 0.204929 | 0.010149 |
| POINT | V5 H+R+F+S | 0.597673 | 0.204867 | 0.006939 |
| GOAL | V0 GOAL17 | 0.395429 | 0.120500 | 0.006909 |
| GOAL | V1 H | 0.398125 | 0.121210 | 0.007839 |
| GOAL | V2 H+R | 0.395508 | 0.120443 | 0.007466 |
| GOAL | V3 H+F | 0.397889 | 0.121243 | 0.005990 |
| GOAL | V4 H+R+F | 0.395810 | 0.120593 | 0.009698 |
| GOAL | V5 H+R+F+S | 0.395102 | 0.120401 | 0.006865 |
| GOAL | C19 | 0.395502 | 0.120531 | 0.005499 |
| GOAL | joueur lissé | 0.398412 | 0.121114 | 0.005239 |

POINT : V2 est meilleur en moyenne, mais trois folds sur quatre sont défavorables en LL/Brier ; IC corrigé du delta LL `[−0.004200, +0.002045]`, dégradation ECE +0.007010 >0.005. **Rejet**.

GOAL : V5 gagne seulement 0.000326 de LL contre GOAL17 et 0.000400 contre C19, sous le minimum0.0005, avec un fold défavorable et IC incluant zéro. Il bat le joueur lissé sur ce développement : delta LL−0.003309, IC corrigé `[−0.006095, −0.000608]`, quatre folds LL/Brier favorables. Cela ne satisfait pas les autres critères et ne remplace pas une confirmation indépendante. **Rejet**.

## Ce que permettent les contrastes

- H seul n'est pas supérieur aux références. Cela ne démontre pas l'inutilité du niveau historique : N−1 manque dans l'entraînement du premier fold, et N−2 est absent partout.
- R ajouté à H améliore LL/Brier sur les quatre folds pour les deux cibles : delta LL POINT−0.002394, GOAL−0.002617 ; IC corrigés favorables. R contient usage, tirs, TOI/PP et calendrier : on ne peut isoler l'effet PP.
- F ajouté à H : delta LL POINT−0.000574, GOAL−0.000236, IC incluant zéro. F ajouté à H+R : +0.000458 / +0.000302, également incertain. Pas de preuve suffisante pour supprimer une feature de production ; 5 et10 matchs n'ont pas été séparés.
- S ajouté à H+R+F : delta LL POINT+0.000044, GOAL−0.000707. Les IC incluent zéro ; celui de GOAL monte à+0.000028. Ne pas arrondir cette borne à zéro pour retenir S. Aucun effet individuel PP/PK/pénalités n'est établi.

AUC, AP, top décile, reliability bins, résultats F/D et tous les folds/IC : `point_results.json`, `goal_results.json`. Aucun modèle défenseur ni ROI évalué.

## Limites et reprise

Deux saisons source, une seule saison OOF ; N−1 non apprenable dans le premier fit. Le dernier fold pèse 19 147 observations : un gain global peut masquer les autres folds. Les agrégats officiels sont rétrospectifs ; aucune archive PP1 ou composition pré-match n'est revendiquée.

Calibration : politique temporelle commune inchangée, jamais apprise sur le fold évalué. Les bins et probabilités sont conservés. Une collision de nom dans les métadonnées a remplacé les diagnostics de sélection du calibrateur des folds0–2 par les bins ; leurs méthodes retenues ne sont donc pas documentées. Les calibrateurs du dernier fold sont sérialisés. Aucun fit répété pour reconstituer ces diagnostics. Cela ne change ni les probabilités ni les critères appliqués.

Les gardes prospectifs conservent la fenêtre2026-11-01–2027-04-20 exclusive, minimum90dates/20000participations/3000défenseurs, archive avant puck drop et ouverture unique. **Non activés : aucun candidat admissible.** Aucun résultat prospectif observé.

293 tests passent avec `-W error`. Le correctif partagé concerne seulement la concaténation avec historique vide ; les calculs historiques sont AST-identiques au cache préexistant. Gros artifacts sous `outputs/structural_20261012/`, hors Git, hashes versionnés. `CLOSED.json` fixe la conclusion. Prochaine action : examiner ces résultats avant tout nouveau protocole ; aucune nouvelle expérience lancée ici.

```bash
.venv/bin/python -m pytest -q -W error
# Restitution seulement, aucun fit :
.venv/bin/python scripts/summarize_structural.py --directory outputs/structural_20261012/point --output /private/tmp/henachel-point-structural-summary.json
.venv/bin/python scripts/summarize_structural.py --directory outputs/structural_20261012/goal --output /private/tmp/henachel-goal-structural-summary.json
# Le runner et ses caches OOF sont identifiés par SHA ; ne pas recréer une autre
# sortie pour relancer l'étude close. Le protocole demeure inchangé.
```
