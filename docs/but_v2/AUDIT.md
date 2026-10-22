# Audit BUT V2 — 20 octobre 2026

Référence : `main` 8fd7a6e. État initial propre, branche utilisateur
`feature/prospective-context` préservée. Aucune instruction AGENTS.md trouvée.

## Architecture constatée

POINT : `model/02_train_point_model.py`, `03_calibrate_point_model.py`,
`05_predict_upcoming_games.py`, `henachel/point.py`. HGB profondeur 4,
300 itérations, taux .05, feuilles 50, L2=1, seed 42. Whitelists explicites;
features décalées dans `henachel/features.py`; calibration temporelle dans
`henachel/calibration.py`. Variables : tirs, TOI/PP, fréquence de points,
repos, séries, retour d'absence, équipe/adversaire et contexte classement.
Les défauts imputés dans le constructeur partagé ne sont pas tous des observations.
Aucune modification de ces fichiers pour BUT V2.

BUT V1 opérationnel : `henachel/goal_operational.py` réentraîne GOAL17
quotidiennement; même famille HGB, pondération de classe positive,
15 variables de `goal.BASE_FEATURES` et 2 disettes. 85% des dates pour fit,
15% pour calibration; choix raw/sigmoid/isotonic sur subdivision temporelle.
Le modèle gelé de février 2024 est un autre comparateur, pas le modèle
opérationnel quotidien. Les probabilités brutes pondérées ne sont pas calibrées.
Les disettes sont des variables apprises, pas un bonus mécanique.
`goal_experiment.py` contient déjà une étude; son holdout a été consommé.
Le rapport du 4 octobre conclut à l'absence de supériorité démontrée sur la
fréquence joueur lissée. Il ne faut pas présenter sa réutilisation comme un test neuf.

Sources : API NHL, boxscores et play-by-play, statistiques PP,
classements pré-match, compositions publiques prospectives, Unibet structuré.
`06_match_model_to_unibet_odds.py`, `07_build_daily_bets.py` : contrats
marché/identité/cote, filtres et recommandations POINT.
`rank_goal_odds.py` classe séparément BUT expérimental.
`08_publish_to_google_sheet.py` publie POINT; `09_settle_previous_bets.py`
et `henachel/settlement.py` soldent le registre canonique.
`collect_prospective_pp.py`, `archive_prospective_outcomes.py`,
`seal_prospective_archive.py` conservent les captures SQLite et résultats.
Les sauvegardes d'artefacts et releases sont distinctes des CSV post-match.

## Workflows et vérifications distantes

Le workflow global rafraîchit les sources, solde, construit POINT, produit BUT
optionnel, archive, puis publie selon le runtime. BUT comporte des étapes
`continue-on-error`; un succès global ne prouve donc pas son succès individuel.
Le dispatcher pilote les déclenchements quotidiens; ne pas le modifier ici.
Le workflow de recherche postgame prévoit 04:30 et 09:00 UTC. L'API Actions
montre des déclenchements planifiés effectivement retardés et ne démontre pas
que deux captures ont eu lieu chaque jour. Le 20 octobre, run 39401645145
créé à 09:34:18Z réussi; le 19, runs 39235339387 (10:08:58Z) et
39281946910 (15:58:47Z) réussis. La cadence prévue n'est pas une garantie.
Le run global 39392942670 du 20 octobre est réussi. Aucun workflow déclenché
ni paramètre de production changé pour cette mission.

## Archives et qualité

`docs/postgame/results`, `longitudinal/{cumulative_summary.json,daily_metrics.csv,
signal_metrics.csv}` examinés. Recalcul local à partir des résultats : résumé
JSON strictement identique. CANONICAL depuis le 17 : 33 POINT, 32 soldés,
15 gagnants, Brier .265851, log-loss .726610; aucun GOAL. CLASSEUR GOAL :
8/27 soldés, un non résolu. EXTRA GOAL : 0/1. Ne pas fusionner ces populations.
Le suivi canonique de picks POINT ne mesure ni tous les joueurs ni BUT.
Le registre SQLite prospectif est une source supplémentaire à auditer avant
une évaluation prospective BUT, pas une autorisation de reconstruire des picks.

Défauts confirmés et corrigés dans la proposition isolée :
- ID NHL utilisé en premier dans l'équipe; aucun repli nominatif sur ID absent.
- Échec/absence de play-by-play et tirs boxscore absents restent inconnus.
- Tentatives de shootout exclues; horodatage de collecte et URLs ajoutés.
- Doublons d'identité rejetés; résumé journalier inclut CANONICAL.
- Brier/log-loss/ROI exigent groupe CANONICAL et horodatages UTC pré-match
  valides, même si une chaîne de provenance prétend le contraire.
- Champ supplémentaire `settlement_status` WIN/LOSS/UNRESOLVED; ancien
  `status=SETTLED` conservé pour compatibilité. Absence ne signifie pas LOSS.
  VOID nécessite une preuve d'annulation/règle bookmaker; pas de VOID inventé.

Les archives versionnées ne sont pas réécrites. Stützle conserve le challenge
NOT_CORROBORATED, POINT gagnant et BUT perdant. Aucun nouvel élément officiel
recherché ne justifie de changer cette revue.

## Données pour la recherche

Snapshot local `data/final/base_features_context_v2.csv` : 104284 lignes,
149 colonnes, du 2024-10-04 au 2026-10-13. Les colonnes brutes permettent de
reconstruire tirs/TOI/PP, buts, repos et contexte antérieurs; disponibilité
historique exacte des corrections NHL inconnue. Le décalage chronologique
empêche d'utiliser le match cible mais ne prouve pas un historique point-in-time.
Ces replays sont exploratoires, jamais des prédictions réellement archivées.

Pas d'historique vérifié dans ce snapshot pour tentatives complètes, distances,
angles/rebonds, xG individuel, compositions PP1/PP2, partenaires de ligne,
gardien projeté/confirmé ou déplacements réels. Les données post-match récentes
ne peuvent pas remplacer leurs valeurs pré-match. Les PP TOI observés ne sont
pas une composition PP1. La distance n'est pas un xG.

## Priorités

P0 : identité, inconnus, preuve temporelle et archives immuables; tests de
non-régression, conservation des populations. P1 : replay commun V1/V2 avec
incertitude appariée par date, protocole fixé avant calcul; mode shadow séparé.
P2 : accumuler captures contextuelles datées, cotes et résultats officiels,
puis valider hors des périodes déjà explorées. P3 : enrichissements gardien,
xG, lignes et fatigue seulement avec couverture/provenance suffisantes.

Validation initiale : 339 tests globaux réussis (52.26 s). Après corrections :
15 tests postgame réussis. Limites restantes : API potentiellement partielle,
corrections officielles tardives, artefact cherché parmi seulement 20 runs
réussis, absence de manifest brut des réponses dans l'ancien collecteur,
biais de sélection des picks et très petits effectifs récents.
