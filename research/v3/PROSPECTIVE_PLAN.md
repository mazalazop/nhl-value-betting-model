# Proposition V3 distincte — non activée, T0 non fixé

Cette proposition n'amende pas A01. Aucun résultat déjà étudié ne sera confirmatoire pour V3. Les captures et périodes examinées dans ce chantier, y compris les sources de novembre 2026, sont exploratoires. Après décision humaine, inscrire un T0 futur et une première date admissible postérieure à toutes les captures déjà examinées ; enregistrer la préinscription et l'activation **avant** T0 et avant les prédictions concernées. Un retard impose un nouveau T0, jamais une antidatation.

## Candidat raisonnable à poursuivre

BUT : `volume_quality`, logistique L2 C=1 + sigmoid, avec liste exacte de variables dans `quality.json`/`run_quality.py` (volume, utilisation, passé offensif, contexte simple, calendrier, disettes/séries, tentatives et proxys géométriques). Ne pas ajouter le déficit de conversion : son apport n'est pas démontré. Ce choix est une sélection exploratoire à valider sur de nouvelles données ; il ne vaut pas supériorité face à V1.

Avant lancement, figer empreintes du code, environnement, liste de variables, configuration, modèle ajusté, calibrateur, historique d'entraînement et manifeste de sources. Ajuster une fois avant T0, avec 15 % dernières dates antérieures réservées à la calibration ; conserver coefficients et calibrateur pendant l'étude. Les variables historiques continueront d'être calculées uniquement avec les observations déjà disponibles avant chaque capture. Un nouveau modèle ou entraînement sera une nouvelle version et une autre cohorte. La référence sera BUT V1 opérationnel, dans la version identifiée à T0 ; changement de version en cours d'étude → pause d'inclusion et revue, sans mélange silencieux.

POINT : aucun remplacement de modèle n'est actuellement justifié. Priorité à une cohorte d'observation exhaustive des cotes et de l'univers candidat. Un filtre `p > 1/cote`, cote ≥1,40, est un comparateur économique possible avec prédicteur inchangé. Les 14 picks du replay ne permettent pas de dimensionner sérieusement une étude de rentabilité. Ni seuil 1,40 ni marge ne seront optimisés sur les performances futures. Un essai économique confirmatoire exigera un protocole séparé et une estimation de variance sur un pilote déclaré exploratoire ; aucune rentabilité n'est promise.

## Effectifs fondés sur l'incertitude

Pour un seul candidat/marché avec deux critères, proposer des gains utiles de **0,001 Brier et 0,003 log-loss**. Il s'agit d'exigences d'utilité proposées à validation humaine, pas de gains observés. Dimensionnement de détection d'une différence non nulle :

`dates requises = dates observées × [(z(1−0,025/2)+z(0,90)) × SE_blocs / gain]^2`.

Le seuil par critère est 0,025 (deux critères), puissance marginale 90 % ; sous ces hypothèses, la borne d'union donne au moins 80 % pour les deux détections. La variance provient des différences appariées par blocs de sept dates. Cela ne garantit pas que l'estimation dépassera exactement le gain utile ; la promotion sera plus exigeante qu'un simple test de non-nullité.

Pour BUT `volume_quality`, il faut environ **244 dates évaluables et 51 612 participations appariées**, à composition identique à l'historique. Gains divisés par deux : 974 dates ; doublés : 61 dates. Pour POINT, le même candidat demanderait environ 811 dates aux gains centraux : peu convaincant comme priorité. Voir `results/power_sensitivity.csv`, qui contient aussi la sensibilité du logit principal. Ce sont des estimations fragiles à la dérive de saison, aux absences et aux cotes manquantes, pas une garantie de puissance.

Proposition BUT : observation jusqu'à au moins 244 dates et 51 612 participations, probablement au-delà d'une saison NHL, avec plafond administratif de trois saisons. Arrêt au plafond sans effectif suffisant → inconcluant. Bilan J30 uniquement qualité de collecte et descriptif ; aucune promotion ni optimisation à cette étape. La durée de 180 jours d'A01 reste intacte et n'est pas réutilisée arbitrairement pour V3.

## Population, capture et analyse

- Toutes les identités joueur-match de la capture V1 admissible, sans sélection après résultat ; mêmes lignes V1/V3. Capture UTC après T0, avant début archivé et avant tout match déjà commencé. Les variables doivent avoir une source disponible à cette heure.
- Couverture de prédiction rapportée sur toutes les identités V1 éligibles, y compris celles sans données V3. Proposer ≥98 % comme garde de qualité, pas comme filtre après résultat. Un défaut doit être visible, pas effacé en réduisant le dénominateur.
- Participation connue avec résultat, non-participation corroborée, participant sans résultat, participation inconnue, match en attente : catégories distinctes et preuves complètes requises. Les absents ne sont jamais des défaites. Maturité : final officiel ou début archivé +12 heures ; match en attente reste en attente.
- Rapporter séparément couverture d'information des lignes matures et disponibilité des résultats parmi participants identifiés, avec garde proposée ≥98 % pour chacune. Garder effectifs exacts, causes d'exclusion, joueurs sans historique et décalage des sources. Toute nouvelle règle demande validation préalable.
- Brier et log-loss : mêmes participants réglés et mêmes dates. Différences appariées par blocs entiers de sept dates, matchs conservés ; 50 000 tirages, deux critères corrigés. Contrôles prévus par saison, fenêtre, domicile/extérieur, décile, TOI et couverture ; sous-groupes descriptifs, sans choix du meilleur segment.
- Top5/10 figés à la capture, ordre probabilité décroissante puis identifiants match/joueur ; jamais remplacer un absent ou un non-réglé. Rapporter sélectionnés, réglés, absents, inconnus et réussite parmi réglés. Ils restent secondaires.
- Simulation économique seulement avec cote pré-match réellement archivée et ≥1,40, une unité par sélection, aucun réinvestissement. Préinscrire le choix Unibet puis FDJ comparable, les règles de void et de sélection avant collecte. Rapporter cotés/non-cotés, concentration, drawdown et incertitude par date. Aucune inférence de profit depuis un score probabiliste.

## Promotion, arrêt et reprise

Aucune promotion automatique. Pour une proposition humaine : deux pertes avec borne supérieure ajustée <0, gains ponctuels au moins 0,001/0,003, calibration et robustesse conformes aux gardes proposées : augmentation de l'ECE à dix intervalles fixes ≤0,01 et augmentation de log-loss ≤0,01 dans chaque segment préspécifié contenant au moins 500 participations (saison, domicile/extérieur, tiers de TOI historique). Ces gardes descriptives représentent des tolérances opérationnelles proposées, non des seuils optimisés ni des preuves d'équivalence statistique ; elles devront être acceptées avant T0. Tout segment trop petit reste inconcluant et demande revue. Couverture conforme. Cette proposition complète reste désactivée et soumise à validation humaine.

Interrompre les nouvelles inclusions en cas de fuite temporelle, collision d'identité, modèle modifié ou échec de conservation ; conserver les preuves et les observations antérieures. Reprendre après correction et validation des empreintes, sans recréer l'étude ni changer les probabilités. Toute correction officielle contradictoire attend une décision humaine tracée, versions conservées. Ne pas arrêter sur un rendement temporairement favorable et ne pas relancer un compteur après un résultat défavorable.

Le prochain accord scientifique devra désigner le candidat, figer toutes les marges ci-dessus et autoriser une PR de raccordement/activation séparée. La présente PR livre uniquement recherche et proposition, avec `enabled=false` et aucun T0.
