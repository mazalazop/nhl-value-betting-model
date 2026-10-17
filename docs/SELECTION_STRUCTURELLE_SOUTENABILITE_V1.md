# HENACHEL — Audits structurels POINT et BUT avant sélection (V1 exploratoire)

Décision métier du 17 octobre 2026 : appliquer à **tous** les candidats finaux un contrôle de soutenabilité de la production, y compris lorsqu'ils ont une forte probabilité brute du modèle.

## Séparation obligatoire

1. **P(HENACHEL)** : valeur brute calibrée de POINT, inchangée (ne pas la recalculer à partir du ressenti).
2. **Audit structurel** : niveau de référence, évolution du rôle, tirs, TOI, PP, séries et rebonds. Diagnostics **qualitatifs**.
3. **Contexte du match** : gardien probable/confirmé, matchup défensif, B2B, fatigue, blessures et composition PP1 récente vérifiée.
4. **Décision de pari** : cote Unibet **>= 1,40** avec horodatage et source; pas de pari si cote inconnue. Un candidat sans cote peut rester dans le classement **probabiliste**.

Aucune de ces étapes ne doit être confondue avec une probabilité issue du modèle. Le modèle inclut déjà des features de séries et des écarts récents : **ne pas appliquer mécaniquement une deuxième pénalité aux mêmes signaux**.

## Audit systématique pour chaque joueur et marché

- Référence : **saison précédente, saison régulière seulement** (si possible N-2 avec poids inférieur), moyenne de points/match, fréquence de matchs à >=1 point, buts/match, tirs, TOI, minutes PP. Marquer explicitement les références fragiles (<30 matchs).
- Saison courante : mêmes métriques, sans les matchs de présaison ni de playoffs.
- Trois et cinq derniers matchs : examiner les séquences individuelles mais réserver leur poids principal à l'identification de changements de **rôle et d'opportunité**, pas à l'extrapolation de la production.
- Décomposer la progression : augmentation des tirs, des tentatives de tirs dangereuses si disponibles, du TOI, du PP TOI, du PP share, du placement et des partenaires; rapprocher cette hausse de celle des points. **Le PP1 doit être observé / vérifié, jamais déduit uniquement d'un temps PP élevé.**
- Risque **hot streak** : production élevée par rapport à la référence, série consécutive et éventuel excès de rendement **non suffisamment expliqué** par le volume et le rôle. Dégrader qualitativement; une série n'a aucune obligation de s'arrêter au prochain match.
- Potentiel **rebond** : sous-production relative au niveau historique **avec** tirs, temps de jeu et rôle maintenus ou en hausse; jamais « il n'a pas marqué, donc il va marquer ».
- Traiter individuellement les risques de gardien / défense adverse / contexte offensif et la cote; ne pas confondre **probabilité de marquer un point** avec **value du pari**.

## Application au marché BUT

La même revue est **obligatoire** avant un Top BUT, mais les références changent :
buts/match, buts/60, tirs et tirs/60, tirs dangereux/xG individuels lorsqu'ils sont fiables,
pourcentage de tirs convertis, temps sur la glace, PP TOI/PP share et position PP1.

- Une série de buts sur très peu de tirs déclenche une vigilance de conversion non durable.
- Une disette de buts n'est un rebond plausible que si les tirs et occasions dangereuses restent élevés.
- Un audit automatisé BUT indépendant est maintenant exécuté **après** le calcul GOAL, uniquement quand celui-ci réussit et est activé.
- Le diagnostic BUT vérifie buts/match, buts/60, tirs/match, tirs/60, taux de conversion, TOI, temps PP, série de buts et disette **relatifs à la référence N-1**. Les tirs dangereux et le statut PP1 sont explicitement inconnus si les sources ne permettent pas de les établir.
- La surcouche est `research_only`, ne change jamais `P(Henachel GOAL)`, ne publie ni ne sélectionne de paris, et conserve le statut du modèle `research_only_operational`.
- Les seuils d'alerte sont des heuristiques non validées. Une hausse de buts soutenue par davantage de tirs ou une promotion de rôle n'est pas dégradée automatiquement.

## Automatisation de diagnostic, sans impact silencieux

`scripts/audit_player_sustainability.py` est une **surcouche indépendante**. Après le calcul POINT pour une date NHL, elle compare :
- `data/raw/stats.csv` (saisons régulières uniquement et dates strictement antérieures au jour ciblé);
- `data/raw/pp_stats_game.csv` quand disponible, pour **minutes PP mesurées**, pas pour un PP1 supposé;
- `outputs/predictions_upcoming_point_enrichi_calibre_v2.csv`.

Elle produit `outputs/player_sustainability_point.csv` avec :
`p_henachel`, `rang_henachel` **inchangés**, moyennes historiques et récentes, volume, PP, séries, diagnostic et justification. Les alertes exploratoires :
- `SURCHAUFFE FORTE` : nette surproduction saisonnière, trois et cinq derniers matchs extrêmement supérieurs à la référence;
- `VIGILANCE SERIE` : cinq matchs consécutifs productifs avec surproduction, ou série plus courte et saison en sur-rendement;
- `REBOND A EXAMINER` : au moins trois matchs sans point, production saisonnière en baisse, tirs et TOI récents maintenus/augmentés;
- `PAS DE SIGNAL FORT`;
- `DONNEES INSUFFISANTES`.

**Ces seuils ne sont pas validés prédictivement.** Ils ne changent ni les probabilités calibrées, ni les picks publiés, ni l'historique de pari. Un analyste confronte ensuite l'alerte au rôle réel, au PP1 vérifié et à l'adversaire. Un jugement qualitatif ne crée pas automatiquement de points de pourcentage.

## Export BUT automatisé

`scripts/audit_goal_sustainability.py` produit `outputs/player_sustainability_goal_research.csv`, sans modifier le fichier de prédictions `outputs/predictions_upcoming_goal_research.csv`.

- **Input modèle** : `probability_goal`, `goal_probability_rank` et `model_status=research_only_operational`, strictement pour la date cible.
- **Historique** : uniquement saison régulière (`id_match` NHL de type `02`) et `date_match < date cible`. Les matchs de présaison, playoffs et les matchs à venir sont exclus.
- **Référence / signaux** : saison précédente et début de saison courant, fenêtres des 3 et 5 derniers matchs, volume tirs/60, buts/60, conversion buts/tirs, temps PP observé, séries de buts et disettes.
- **Diagnostics** : `SURCHAUFFE CONVERSION`, `SERIE DE BUTS A EXAMINER`, `REBOND A EXAMINER`, `VOLUME EN HAUSSE`, `PAS DE SIGNAL FORT` ou `DONNEES INSUFFISANTES`.
- **Sécurité** : le PP1 n'est jamais déduit de simples minutes PP ; la probabilité GOAL et son rang sont recopiés **sans recalcul** ; aucune donnée post-match ni donnée d'une autre date n'est acceptée ; l'étape n'est pas bloquante pour POINT ni pour GOAL.
- **Validation** : tests unitaires et première exécution hors pipeline sur l'archive du 17/10/2026. Tout impact automatique sur le ranking exige une évaluation prospective/walk-forward.

**Attention :** le fait qu'une série de buts paraisse improbable à prolonger ne signifie pas que sa probabilité diminue mécaniquement au match suivant. Les alertes sont des points de contrôle humain, pas un modèle de régression supplémentaire.

## Exemple du 17 octobre 2026

- Seth Jones : référence ~0,62 point/match, 1,80 sur les 5 derniers, 2,67 sur les 3 derniers. Le volume et le PP ont également augmenté : **surperformance partiellement soutenue**, mais forte vigilance.
- Mitch Marner : cinq matchs consécutifs avec point; le rôle PP1 est revenu mais la production récente est sensiblement supérieure à sa référence. **Vigilance**.
- Nick Schmaltz : points/match en hausse mais TOI saisonnier légèrement inférieur; tirs/PP renforcés. **Vigilance sans exclusion**.
- Lucas Raymond : trois matchs sans point et production en retrait malgré davantage de tirs et de TOI. **Rebond à examiner**.
- Jack Eichel : total saisonnier élevé, mais cinq derniers matchs revenus près de la référence structurelle. **Pas de pénalité de série automatique**.

## Mise en production d'une pénalité quantitative : pas avant validation

Une future modification du classement automatique exige :
- walk-forward sur saisons historiques, dates pré-match et informations réellement accessibles à l'époque;
- ablation de l'alerte et comparaison à POINT seul : Brier, log-loss, calibration par segment, taux de réussite et ROI **hors échantillon** aux cotes historiques disponibles;
- mesure séparée des signaux pour éviter le double comptage des streaks déjà présents dans POINT;
- validation explicite de nouveaux seuils avant tout impact sur les recommandations publiées.

Cette V1 ajoute le **contrôle méthodologique obligatoire** et l'export diagnostique, pas une fausse nouvelle probabilité ou une promesse de meilleure performance.
