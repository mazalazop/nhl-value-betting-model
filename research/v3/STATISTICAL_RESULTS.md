# Résultats statistiques V3 — exploration, aucune promotion

Les fichiers JSON/CSV de `results/` contiennent tous les effectifs, fenêtres exactes, segments, méthodes et intervalles. Cette étude réutilise des périodes historiques : elle ne constitue pas un nouveau test prospectif. Une perte plus faible est meilleure. Les intervalles sont des bootstraps appariés par blocs circulaires de sept dates, conservant les matchs entiers ; 50 000 réplications, correction de Bonferroni sur 128 comparaisons métriques (64 contrastes × deux pertes). Les extrêmes corrigés reposent sur peu de tirages et restent des approximations, sans protection contre les biais de reconstruction historique.

Le protocole exploratoire initial a été commité avant les ajustements. Le calcul initial d'incertitude du replay principal (4 000 tirages/famille 28) a été remplacé par `resummarize.py` après correction du décompte des comparaisons, avant inspection des résultats agrégés. Les prédictions n'ont pas été refaites pour ce changement. `main.json` conserve le manifeste initial et le manifeste d'inférence corrigé. L'expérience tirs a utilisé directement 128/50 000. Ce journal ne doit pas être présenté comme une préinscription prospective.

## Références et chronologie

Références : algorithmes POINT et GOAL17 existants, inchangés, réentraînés à chaque fenêtre. Candidats : régression logistique L2 C=1, imputation et standardisation ajustées sur entraînement, calibration sigmoid sur le passé ultérieur. Les 15 % dernières dates précédant chaque fenêtre sont réservées à la calibration. Les variantes raw/sigmoid/isotonic sont toutes publiées, sans choisir le vainqueur après test. Aucune recherche d'hyperparamètres additionnelle.

Le replay principal utilise sept fenêtres et 182 445 participations communes par marché sur 819 dates. L'expérience tirs utilise cinq fenêtres, 58 169 participations par marché et 275 dates, sur deux saisons. Le dernier match historique du test principal précède le 17 octobre 2026. Voir les limites train/calibration/test exactes dans les deux JSON.

Les taux Top5/10 sont calculés chaque jour parmi les participants historiques ; égalités départagées par match puis identifiant NHL. Ils ne sont pas les Top10 prospectifs figés. `ranking_uncertainty.csv` fournit des intervalles secondaires à 95 % par blocs de dates pour les comparateurs principaux. Ils ne sont pas ajustés aux comparaisons multiples et restent descriptifs, sans test confirmatoire de supériorité. La couverture de 100 % indique uniquement l'appariement dans cet univers. Le ROI est indisponible faute de cotes pré-match pour cet univers.


## Comparaison principale

| Marché | Variante | N | Brier | Log-loss | Top5 | Top10 | Δ log-loss vs référence [IC ajusté] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BUT | reference | 182445 | 0.120473 | 0.395153 | 36.95% | 34.93% | Référence |
| BUT | reference_sigmoid | 182445 | 0.120473 | 0.395153 | 36.95% | 34.93% | +0.000000 [+0.000000; +0.000000] |
| BUT | reference_isotonic | 182445 | 0.120569 | 0.396888 | 36.65% | 34.48% | +0.001735 [+0.000824; +0.002814] |
| BUT | full_sigmoid | 182445 | 0.121293 | 0.397504 | 35.75% | 33.88% | +0.002352 [+0.000430; +0.004685] |
| POINT | reference | 182445 | 0.204448 | 0.596527 | 68.47% | 64.76% | Référence |
| POINT | reference_sigmoid | 182445 | 0.204430 | 0.596125 | 68.42% | 64.52% | -0.000402 [-0.001100; +0.000038] |
| POINT | reference_isotonic | 182445 | 0.204620 | 0.597825 | 68.08% | 64.33% | +0.001298 [+0.000425; +0.002470] |
| POINT | full_sigmoid | 182445 | 0.204591 | 0.596670 | 65.81% | 63.35% | +0.000143 [-0.002317; +0.002856] |

## Expérience tirs : volume et proxys

| Marché | Variante | N | Brier | Log-loss | Top5 | Top10 | Δ log-loss vs référence [IC ajusté] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BUT | reference | 58169 | 0.120949 | 0.396162 | 35.13% | 33.67% | Référence |
| BUT | no_volume_no_quality | 58169 | 0.122123 | 0.400054 | 31.85% | 31.93% | +0.003892 [+0.000998; +0.007873] |
| BUT | volume | 58169 | 0.121733 | 0.398634 | 34.25% | 33.20% | +0.002473 [-0.000161; +0.006331] |
| BUT | quality | 58169 | 0.121131 | 0.396447 | 33.82% | 32.44% | +0.000285 [-0.002377; +0.003582] |
| BUT | volume_quality | 58169 | 0.121100 | 0.396301 | 34.47% | 32.91% | +0.000139 [-0.002427; +0.003361] |
| BUT | volume_quality_conversion_gap | 58169 | 0.121180 | 0.396436 | 34.84% | 33.05% | +0.000274 [-0.002295; +0.003613] |
| POINT | reference | 58169 | 0.204935 | 0.597536 | 66.62% | 63.56% | Référence |
| POINT | no_volume_no_quality | 58169 | 0.205359 | 0.598449 | 64.15% | 61.45% | +0.000913 [-0.002547; +0.005461] |
| POINT | volume | 58169 | 0.204951 | 0.597528 | 64.95% | 61.82% | -0.000008 [-0.003439; +0.004433] |
| POINT | quality | 58169 | 0.204595 | 0.596671 | 64.29% | 61.89% | -0.000865 [-0.004221; +0.004022] |
| POINT | volume_quality | 58169 | 0.204553 | 0.596578 | 63.78% | 61.56% | -0.000958 [-0.004267; +0.003837] |
| POINT | volume_quality_conversion_gap | 58169 | 0.204903 | 0.597372 | 64.00% | 61.78% | -0.000164 [-0.003821; +0.004930] |

Le candidat principal ne démontre aucun gain POINT. Pour BUT il dégrade les deux pertes par rapport à la référence, y compris avec les intervalles ajustés. Il ne doit pas remplacer le modèle actuel.

Dans l'expérience tirs, `quality` désigne un **ensemble de tentatives totales et de proxys géométriques/rebonds**, pas un xG ni un test isolé de qualité pure. `volume` désigne les tirs cadrés et tirs cadrés/60. L'ajout de l'ensemble au candidat BUT réduit la log-loss de 0,002334 [−0,003346 ; −0,001286], mais la différence finale avec la référence vaut +0,000139 [−0,002427 ; +0,003361] : inconcluant. POINT présente un gain descriptif de −0,000958 contre la référence, avec intervalle [−0,004267 ; +0,003837] : également inconcluant. L'apport des tirs cadrés une fois l'ensemble ajouté n'est pas établi. Impossible d'en conclure que les xG surpassent le volume.

## Ablations du candidat principal

Différence = modèle privé de la famille moins modèle complet. Un nombre positif signifie que retirer la famille détériore le candidat. Ces ablations sont conditionnelles à cette architecture ; elles ne mesurent pas l'effet causal d'une promotion PP1 ou l'importance exacte dans le modèle de production.


| Marché | Famille retirée | Δ log-loss | IC ajusté | Lecture |
| --- | --- | --- | --- | --- |
| BUT | calendar | +0.000095 | [-0.000084; +0.000271] | Non établi |
| BUT | context | +0.000174 | [-0.000018; +0.000367] | Non établi |
| BUT | drought | +0.000376 | [+0.000097; +0.000702] | Apport dans ce candidat |
| BUT | hot_streak | -0.000014 | [-0.000061; +0.000028] | Non établi |
| BUT | offense | +0.001391 | [+0.000573; +0.001996] | Apport dans ce candidat |
| BUT | role_change | +0.000004 | [-0.000013; +0.000020] | Non établi |
| BUT | usage | +0.002402 | [+0.001526; +0.003477] | Apport dans ce candidat |
| BUT | volume | +0.001574 | [+0.000948; +0.002296] | Apport dans ce candidat |
| POINT | calendar | +0.000268 | [-0.000030; +0.000553] | Non établi |
| POINT | context | +0.000403 | [-0.000078; +0.000881] | Non établi |
| POINT | drought | +0.000230 | [-0.000005; +0.000498] | Non établi |
| POINT | hot_streak | -0.000005 | [-0.000036; +0.000026] | Non établi |
| POINT | offense | +0.001769 | [+0.000890; +0.002458] | Apport dans ce candidat |
| POINT | role_change | -0.000006 | [-0.000022; +0.000012] | Non établi |
| POINT | usage | +0.003412 | [+0.002288; +0.004952] | Apport dans ce candidat |
| POINT | volume | +0.000924 | [+0.000378; +0.001550] | Apport dans ce candidat |

L'historique de production individuelle, l'utilisation (TOI et PP ensemble) et le volume de tirs apportent de l'information aux deux candidats. La famille disette contribue au candidat BUT, mais elle combine disettes POINT et BUT : cela ne démontre ni un bonus de disette, ni son signe causal, ni un gain du candidat sur V1. Les séries chaudes n'apportent pas de gain indépendant établi. Le calendrier et le contexte adverse simplifié restent inconcluants sur les deux critères conjointement.

Les variables « changement de rôle » sont ici TOI3−TOI10 et PP3−PP10, combinaisons déjà représentées par les fenêtres d'utilisation. Leur ablation dans une régression linéaire ne permet pas de juger l'utilité de véritables nouvelles compositions. Aucun gardien confirmé, PP1 ou xG point-in-time n'a été testé.

Le déficit de conversion compare les buts des dix matchs antérieurs aux tirs multipliés par le taux individuel **antérieur à ces dix matchs**, avec au moins 50 tirs plus anciens. Ce n'est pas un déficit de xG. Son ajout ne démontre aucun gain. `conversion_strata.csv` donne des groupes descriptifs après analyse (déficit <−1 but, écart ±1, surplus >1, inconnu), avec volume/TOI/PP et scores ; ils ne sont pas de nouvelles règles de sélection ni un résultat causal. L'expérience ajustée volume+rôle+proxys constitue le contrôle prédictif pertinent.

## CANONICAL POINT : réussite et rendement

174 sélections réglées sur 15 dates, 99 succès (56,90 %), Brier 0,240063, log-loss 0,673386 ; probabilité moyenne annoncée 61,44 %. Le ROI simulé est **−13,44 %**, soit −23,39 unités pour une unité par sélection ; drawdown de 25,54 unités mesuré sur les soldes journaliers. L'intervalle descriptif par blocs de dates du ROI est [−24,42 % ; −0,85 %], avec seulement 15 dates et une forte fragilité : aucune conclusion confirmatoire.

**151/174 picks avaient déjà une EV négative selon le modèle**. L'EV moyenne annoncée était −6,49 %. La cote moyenne gagnante vaut 1,5213 : 56,90 % × 1,5213 −1 = −13,44 %. Un taux supérieur à 50 % ne suffit pas à ces cotes. Le code de recommandation privilégie la probabilité, puis l'edge ; le marqueur value n'est pas une condition obligatoire. La sélection et le prix sont donc une priorité avant d'ajouter de la complexité au modèle.

La sélection des fortes probabilités est différente de la calibration de tout l'univers. Le replay de la référence est globalement bien calibré (voir figures) et les recalibrations supplémentaires n'établissent pas de gain robuste. La calibration existante est essentielle, notamment après l'apprentissage pondéré ; retirer celle-ci dégrade fortement les probabilités brutes. Cela ne justifie pas de recalibrer rétrospectivement le registre.

Les politiques value fixes sont rejouées sur les dates après sept journées historiques, avec seulement les résultats dont une révision était archivée avant la décision. Le filtre brut edge>0 retient 14 picks sur cinq dates, neuf succès, ROI −1,79 % : beaucoup trop peu pour choisir une politique. Le filtre marge 0,05 n'en conserve qu'un (perdant). La réduction de probabilité de 0,05 est mathématiquement équivalente à cette marge et n'est **pas** une borne de confiance. Les variantes recalibrées, tous les effectifs, segments et concentrations figurent dans `canonical.json`. Ce sont des simulations sur un univers déjà sélectionné ; aucun pari réel ni disponibilité commerciale indépendante n'est affirmé.

CLASSEUR et EXTRA sont publiés séparément dans `groups.json`. Les catégories ne sont jamais additionnées aux 174 CANONICAL. Les segments avec un ou quelques picks restent descriptifs, sans classement des joueurs « rentables ».

## Études prospectives existantes : lecture seule


| Étude | Modèle | Archivés | Réglés | Dates | Brier | Log-loss | Top10 gagnants/réglés | ROI Top10 coté |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BUT original | v1 | 3295 | 2605 | 10 | 0.126340 | 0.409555 | 33/94 sur 100 | -19.11% (n=75) |
| BUT original | v2 | 3295 | 2605 | 10 | 0.127821 | 0.415503 | 26/94 sur 100 | -30.69% (n=72) |
| A01 | v1 | 1900 | 1502 | 6 | 0.125210 | 0.406829 | 17/56 sur 60 | -28.96% (n=48) |
| A01 | v2 | 1900 | 1502 | 6 | 0.126196 | 0.411448 | 18/57 sur 60 | -12.92% (n=48) |

Ces études se chevauchent et ne sont pas fusionnées. Les 690 non-réglés du rapport original conservent sa définition historique. A01 distingue 1 502 participations réglées et 398 non-participations corroborées sur 1 900 paires ; aucun inconnu dans cette capture, disponibilité des résultats parmi participants 100 %. L'ancien champ de couverture reste 79,05 %, inchangé. Le mode externe du rapport A01 est PROSPECTIVE ; un sous-champ de couverture porte encore une étiquette exploratoire héritée. Nous le conservons tel quel et utilisons le statut externe, sans réparer l'étude dans cette recherche.

Six dates A01 et dix dates originales ne permettent pas une conclusion de supériorité. Les Top10 sont ceux figés dans les registres, sans remplacement. Les ROI sont des simulations uniquement pour les sélections cotées ; aucun recalcul du protocole de ces études n'est effectué.

## Décision de recherche

Conserver les modèles actuels. Priorité à une collecte exhaustive des probabilités/cotes et à la validation d'une politique value indépendante. Le candidat enrichi de tirs est une piste expérimentale BUT raisonnable, sans supériorité démontrée ; son gain marginal face au logit simple ne suffit pas à le promouvoir. POINT ne justifie pas aujourd'hui le remplacement de son prédicteur. Aucun résultat n'autorise un bonus de disette, un malus de série ou une nouvelle règle PP1.

Les résultats par fenêtre, saison, domicile/extérieur et toutes les ablations sont disponibles dans les JSON ; `comparisons.csv`, `contrasts.csv` et `fold_metrics.csv` permettent leur réanalyse. Les figures `main_calibration.svg` et `quality_calibration.svg` affichent les intervalles contenant au moins 30 observations. Les intervalles vides ne sont pas assimilés à une bonne calibration.
