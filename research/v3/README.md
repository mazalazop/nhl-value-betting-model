# HENACHEL V3 — recherche isolée

**Aucun candidat V3 n'a démontré une supériorité suffisante pour remplacer les modèles actuels.** La recherche identifie deux priorités : mieux sélectionner les cotes POINT et poursuivre l'évaluation des tentatives/proxys de tirs pour BUT. Production, A01, protocole et archives restent inchangés. Candidats désactivés, aucune fusion ou activation requise pour lire les résultats.

- [Inventaire des données et provenance](DATA_AUDIT.md)
- [Résultats chiffrés et ablations](STATISTICAL_RESULTS.md)
- [Collecte pré-match proposée](COLLECTION_PLAN.md)
- [Proposition prospective distincte](PROSPECTIVE_PLAN.md)
- [Reproduction et validations](REPRODUCE.md)
- [Compatibilité avec main au 9 novembre](INTEGRATION_20261109.md)
- [Tableau de tous les comparateurs](results/comparisons.csv), [différences appariées et incertitude](results/contrasts.csv), [fenêtres](results/fold_metrics.csv)

## Réponses aux dix questions

| Question | Réponse et niveau de preuve |
|---|---|
| Variables utiles à POINT ? | Dans le logit exploratoire, historique offensif, TOI/PP et volume améliorent les pertes par ablation. Aucun gain global établi face au POINT actuel. |
| Variables utiles à BUT ? | Mêmes familles ; disettes conjointes POINT/BUT apportent un petit signal dans ce candidat. Le logit principal reste moins bon que V1. |
| Volume ou qualité ? | Les tentatives et proxys géométriques améliorent BUT au-delà des tirs cadrés dans le logit. Ce paquet mélange volume de tentatives et géométrie : il ne démontre pas la supériorité d'un véritable xG. |
| Disettes et hot streaks ? | Pas de preuve d'un bonus automatique. Séries chaudes sans apport indépendant établi ; déficit de conversion récent sans gain démontré. |
| Changements PP1/TOI ? | Utilisation globale informative ; pas de données historiques de PP1 confirmé. Les différences TOI/PP sont déjà représentées par les fenêtres de base. |
| Adversaire et gardiens ? | Contexte simplifié inconcluant ; gardiens non testables faute de snapshots pré-match fiables. |
| Pourquoi POINT perd malgré 56,90 % de réussite ? | Cotes gagnantes moyennes 1,5213 et 151/174 picks à EV négative selon leurs propres probabilités. ROI simulé −13,44 %. |
| Recalibrer, filtrer ou réentraîner ? | La calibration actuelle est indispensable ; une recalibration supplémentaire n'apporte pas de gain établi. La collecte des cotes et l'essai d'un filtre value sont prioritaires, sans promesse de rentabilité. |
| Candidat prometteur ? | BUT logit + tentatives/proxys : gain face au logit simple, comparaison V1 inconcluante. POINT : conserver le prédicteur ; aucune promotion V3. |
| Que collecter maintenant ? | Univers candidat complet, cotes/horaires, variables réellement disponibles ; puis compositions/PP et gardiens sourcés, tirs officiels passés ; xG seulement avec licence et version admissibles. |

## Priorités

| Priorité | Action | Impact potentiel | Difficulté | Preuve actuelle |
|---|---|---|---|---|
| 1 | Capturer toutes les cotes et tous les candidats ; étudier value sans changer la production | Élevé sur la sélection économique | Moyenne | Diagnostic négatif solide sur 174 picks ; profit d'un nouveau filtre non démontré |
| 2 | Conserver calibration et références actuelles | Protection contre une dégradation | Faible | Replays appariés larges ; candidats sans supériorité globale |
| 3 | Archiver tentatives/géométrie officielles avant chaque prévision | Gain prédictif potentiel BUT | Moyenne | Gain marginal historique sur 58 169 lignes, pas de gain établi vs V1 |
| 4 | Capturer compositions/PP1 et gardiens avec heures et confiance | Potentiel non mesuré | Moyenne à élevée | Données historiques insuffisantes |
| 5 | Étudier un fournisseur xG sous licence compatible | Potentiel non mesuré | Élevée | Pas de jeu point-in-time exploitable validé |

Les 182 445 participations du replay principal, les 58 169 de l'expérience tirs, les 174 picks CANONICAL et les registres prospectifs sont des populations différentes. Leurs performances ne se combinent pas. Tous les gains de recherche restent exploratoires. Les tests prouvent des propriétés techniques, pas la rentabilité ni la supériorité prédictive.
