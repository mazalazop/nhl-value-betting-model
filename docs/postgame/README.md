# HENACHEL — Suivi post-match et apprentissage prudent

## Pourquoi ce dispositif existe

Un pari BUT perdu ne signifie pas nécessairement que le joueur a produit peu d'occasions. Inversement, un but sur une tentative peu dangereuse ne confirme pas à lui seul une prédiction de haute qualité. Le but est de suivre **à la fois l'issue officielle et les opportunités réellement créées**, sans confondre a posteriori les statistiques du match avec les variables disponibles avant coup d'envoi.

## Deux populations strictement distinctes

| Groupe | Source | Garantie de sélection pré-match | Métriques |
|---|---|---|---|
| CLASSEUR | Transcription datée des 56 picks discutés pour le 17/10 | Non prouvable à partir du commit rétrospectif seul | Résultat officiel, tentatives, tirs, groupe de critères |
| EXTRA | Stützle POINT/BUT et Chychrun POINT discutés séparément | Ne pas fusionner au classeur ni au bilan de paris | Résultat et contexte descriptifs |
| CANONICAL | `master_daily_bets_history.csv` téléchargé depuis les artefacts Actions | `captured_at` strictement avant `start_time_utc`, identités et cote figées | En plus : Brier, log-loss, retour simulé à une unité |

**Ne jamais additionner ces trois populations** pour prétendre mesurer une performance de pari indépendante. Un même joueur peut apparaître dans deux groupes.

## Contrôle de données

Le collecteur :
- Utilise le boxscore et le play-by-play de l'API officielle NHL ; il refuse de solder des matchs non terminés et vérifie les identifiants de match/joueur s'ils sont fournis.
- Enregistre buts, passes, points, tirs cadrés, tentatives/ratés/contres, temps de glace et PP TOI **uniquement s'il est disponible**. Une donnée manquante n'est pas un zéro.
- Calcule des distances géométriques de tirs comme **proxys non calibrés** (les coordonnées d'un tir bloqué peuvent correspondre au lieu du blocage). Ces proxys **ne sont pas des xG**.
- Ne compte pas les joueurs absents ou non rapprochés comme perdants.
- Déduit le succès officiel POINT ou BUT uniquement des statistiques homologuées NHL, jamais d'une affirmation non vérifiée sur un but annulé.
- Importe le registre de paris avec identifiants conservés, prix et probabilités figés, horodatage UTC strictement antérieur à la mise au jeu, sans reconstruction rétroactive.

## Cas Stützle — 17 octobre 2026

Stützle (Ottawa) termine officiellement **0 but, 1 passe, 1 point** contre Philadelphie. Le pick POINT discuté après le classement est gagnant ; le pick BUT est perdant. Un but annulé sur challenge a été signalé par l'utilisateur, mais **aucun compte rendu indépendant corroborant cet événement spécifique n'a encore été identifié**. L'incident figure en revue `docs/postgame/reviews/2026-10-17.csv` avec statut `NOT_CORROBORATED`. Il ne change ni le résultat officiel, ni les xG, ni les statistiques de tir.

## Première observation, 17 octobre (un seul jour)

- CLASSEUR : POINT 11/27 ; BUT 8/27 ; deux picks Tippett sans feuille de match, non comptabilisés comme des pertes.
- EXTRA : POINT 2/2 (Stützle et Chychrun), BUT 0/1 (Stützle).
- CANONICAL : 11 picks POINT réellement capturés avant coup d'envoi ; 4/11. Brier 0,325498, log-loss 0,852456, simulation fictive à 1 unité par pick −45,09 %. Ce n'est **pas** un ROI de mises effectivement enregistrées, ni un échantillon suffisant pour calibrer le modèle.

## Exploitation au fil des journées

Workflow `henachel-postgame-quality.yml` : deux vérifications quotidiennes et déclenchement manuel. Il récupère le dernier artefact canonique HENACHEL valide, conserve les captures pré-match, réconcilie les boxscores officialisés et calcule :
- `docs/postgame/longitudinal/daily_metrics.csv`
- `docs/postgame/longitudinal/signal_metrics.csv`
- `docs/postgame/longitudinal/cumulative_summary.json`

Les bilans de 7 jours utilisent **les 7 jours calendaires**, et non les 7 dernières journées comportant des picks.

### Critères de promotion scientifique

Après accumulation de plusieurs semaines/centaines d'observations, comparer sur nouvelles dates :
1. Volume de tirs et occasions rapprochées *avant* match versus issue du match suivant, pas les tirs du match déjà joué.
2. Évolution réelle du PP1/PP TOI et du rôle vs simple disette en buts.
3. Calibration du modèle : log-loss, Brier, fréquence par déciles et classement des meilleurs picks, avec évaluation chronologique et bootstrap par date/match.
4. Si disponibles, xG individuels par localisation et gardiens titulaires : sources vérifiables et horodatées.

**Aucune modification des coefficients HENACHEL POINT/GOAL, ni des probabilités ou recommandations, n'est autorisée sur la seule base de la soirée du 17 octobre.** Le modèle BUT reste expérimental.
