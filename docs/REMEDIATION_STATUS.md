# Remédiation locale — état et limites

## Statut

**Terminé avec réserves externes.** La cohérence et les garde-fous sont testés sur fixtures synthétiques. La performance prédictive réelle et le fonctionnement des services externes ne sont pas démontrés par ces tests. Aucun push, aucun pari, aucune écriture dans Google Sheets réel, aucun credential lu ou modifié.

Départ : `c51d544b9b7d7bf49d6b3c39740fffdb8ab35565`. Branche : `astra/audit-remediation`. Environnement Python 3.11.9 ; dépendances installées figées dans `requirements.lock`. Aucun dataset ou artefact modèle de la référence documentaire n'était présent au départ.

## Corrections et preuves

| Bloc | Cause / correction | Tests principaux |
|---|---|---|
| Features 01/05 | Définitions dupliquées divergentes : moteur commun, fenêtres sur les derniers matchs, saisons séparées, `passes_moy_10`, defaults cohérents, suppression d'imputations globales | `test_feature_parity.py`, rapports `feature_parity_before.json` et `feature_parity_after.json` |
| PP/TOI courant | `cumsum() - courant` dépendait de la présence du résultat courant ; NaN pouvait devenir zéro. Décalage avant accumulation, total inconnu si historique incomplet, moyenne sur observations passées disponibles | `test_current_missing_toi_pp_cannot_change_pregame_season_statistics`, `test_unknown_pp_stays_unknown_but_observed_zero_is_zero` |
| PP/standings | Source PP obsolète, absence assimilée à zéro, snapshot mal daté/hors saison : fraîcheur, couverture, saison, date API et jointure réelle contrôlées | `test_pp_standings.py` |
| Intégrité NHL | Score présent assimilé à final, stats absentes converties : statut final requis, clés uniques, points/buts/passes et équipes validés | `test_data_contracts.py` |
| Calibration | Sigmoid différent entre 03 et 05 : logit commun, choix temporel déterministe, fenêtres vides/classe unique explicités, test strictement postérieur | `test_calibration.py`, `test_science.py` |
| Bookmaker | Identités réduites à initiale/dernier mot, ambiguïtés, marché et fraîcheur insuffisants : candidat unique dans les deux sens, équipes/date/heure/marché vérifiés, rejet des NaN d'identité, alias Utah, bet_id basé sur IDs NHL | `test_matching.py` |
| Calculs et picks | Probabilité implicite non contrôlée, hot-streak perdu : recalcul local, validation, propagation du flag, activation explicite ; EV négative toujours autorisée par la règle existante | `test_matching.py`, `test_history.py` |
| Rosters/futur | Dernière apparition seule et métadonnées historiques écrasant le match futur : roster frais prioritaire, rookies/transferts/retours longs couverts, fallback explicite, calendrier futur conservé | `test_rosters_empty.py`, `test_model_entrypoints.py` |
| Historique | Identifiants perdus et écritures fragiles : CSV canonique complet, verrou POSIX, remplacement atomique, odds et résultats existants préservés | `test_history.py` |
| Settlement | Résultats supposés, fallback « over », absence de connexion quotidienne : final requis, POINT 1+ validé, inconnus conservés, corrections NHL et journal à identifiant stable | `test_settlement.py`, y compris interruption après écriture du journal puis double relance |
| Publication | Pending/vides remplaçant win/loss et `clear → write` : vues issues du canonique, garde de migration, batch unique, valeurs littérales, scope Sheets seul | `test_publication.py` |
| Exploitation | Inputs shell interpolés, concurrence, historique éphémère : validation runtime, sérialisation, restauration par branche, nettoyage credentials, snapshots scraper frais | `test_runtime.py`, `test_empty_pipeline.py` |
| Science/reproductibilité | Labels invalides tronqués/filtrés, artefacts insuffisants : contrats stricts, paramètres communs, manifestes et bundles, walk-forward séparé | `test_science.py`, `test_model_entrypoints.py` |
| Performance | Deux téléchargements du même boxscore et rescans de streaks : cache final partagé à TTL, full refresh, calcul incrémental équivalent | `test_cache.py`, `test_streaks.py`, `PERFORMANCE_CHECK.json` |

L'invariance aux labels et aux features postérieurs à la coupure est testée sur le modèle et son calibrateur. Les 81 features demandées sont comparées sur les cas synthétiques de parité, avec et sans standings, incluant deux saisons, fenêtres 5/10/20, valeurs manquantes, absences et changement d'équipe. Ce n'est pas une validation exhaustive sur toutes les rencontres NHL réelles.

Le benchmark synthétique de 1 000 lignes compare les streaks incrémentaux à une référence brute indépendante : environ 0,047 s contre 1,877 s, mémoire Python maximale similaire (~243/247 ko). Ce résultat ne mesure ni la RAM globale ni la durée du pipeline réel. Le test du cache ramène deux appels simulés à un seul ; aucun volume HTTP réel n'a été mesuré.

## Contrat des cotes normalisées

Chaque ligne doit contenir les champs métier déjà attendus par 06, notamment `bookmaker`, `event_id`, `home_team`, `away_team`, `team`, `player_name`, `market=player_points`, `stat=points`, `threshold=1`, une issue explicite parmi `1_plus`, `points_1_plus`, `1+`, `over_0.5`, une cote décimale > 1 et une probabilité implicite cohérente.

Champs temporels requis : `date_match` (date NHL), `event_start_utc` et `captured_at` (horodatages avec fuseau). `event_start_utc` doit correspondre au `start_time_utc` NHL ; cote collectée depuis au plus 6 h, jamais dans le futur, match pas encore commencé. Si `nhl_game_id` est fourni, il doit correspondre. L'identifiant bookmaker n'est pas confondu avec l'identifiant NHL. La tolérance absolue de comparaison avec `1 / odds_decimal` est `1e-6`.

Le scraper externe n'a pas été modifié ni testé ici : la présence et la signification de ces champs doivent être vérifiées avant exploitation. Aucune date de collecte n'est inventée au moment du matching. Les erreurs de schéma bloquent explicitement ; les lignes incompatibles produisent un rejet motivé.

## État des étapes

| Étape | État local | Limite externe |
|---|---|---|
| 00 | Corrigé, parsing/contrats/cache testés | Refresh NHL multi-saisons réel non exécuté |
| 00a | Corrigé, plage et types 2/3 | Couverture API PP réelle non mesurée |
| 00b | Corrigé, statuts et intégrité testés | Boxscores réels non récupérés |
| 00c | Corrigé, snapshots pré-match contrôlés | Disponibilité des snapshots historiques non vérifiée |
| 01 | Parité et anti-leak synthétiques validés | À rejouer sur dataset réel |
| 02 | Entrypoint, fit, exports et bundle testés | Métriques réelles indisponibles |
| 03 | Entrypoint et calibration partagée testés | Référence historique non reproductible sans artefacts |
| 05 | Entrypoint futur, rosters, sorties vides testés | Rosters réels/lineups non vérifiés |
| 06 | Matching et calculs stricts testés | Adaptateur scraper à vérifier |
| 07 | Picks et historique testés | Règle EV négative conservée ; hot-streak optionnel |
| 08 | Mocks, batch et garde d'historique testés | Aucun appel Google réel |
| 09 | Idempotence, corrections et reprises testées ; connecté au workflow | DNP/annulations restent non réglés sans règle bookmaker |
| 10 | Statuts de comparaison corrigés, choix sur test supprimé | Comparabilité de la référence non vérifiée |
| 11 | Walk-forward/metrics/bins testés | Pas de résultat prédictif réel publié |

## Réserves avant exploitation

- Les tests ne prouvent pas une rentabilité ni une amélioration prédictive. Les valeurs Brier 0.205518, log-loss 0.599639, AUC 0.678702 restent documentaires. Le test final ne sélectionne aucune version.
- Le workflow est vérifié statiquement (YAML, shell, contrats), pas exécuté sur GitHub Actions. Son scraper dépend encore d'un runner macOS externe et de son repository ; sa révision n'est pas figée ici. Avant production, fixer une révision vérifiée de ce scraper et tester son contrat réel.
- Les artifacts d'historique sont propres à chaque branche et conservés 90 jours. Une sauvegarde durable indépendante reste nécessaire. Un historique existant dans Sheets doit être rapproché/importé dans le canonique avant publication ; le garde-fou bloque une perte silencieuse. Ne pas amorcer un historique vide pour contourner cette migration.
- Roster actuel ne signifie pas participation garantie au prochain match. Les lineups/scratches ne sont pas une nouvelle source ajoutée dans cette mission. Le fallback historique peut omettre un rookie ou un transfert non documenté ; il est signalé.
- Les corrections NHL anciennes peuvent nécessiter un full refresh des boxscores. Le calendrier et les PP restent rafraîchis largement ; leur coût réel n'a pas été mesuré.
- Les verrous locaux ciblent macOS/Linux. Les styles Sheets sont appliqués après le batch des valeurs : un échec de style peut laisser une vue correcte mais incomplètement formatée ; une relance conserve les résultats.

## Reproduire la validation finale

Depuis la racine, avec l'environnement installé :

```bash
bash scripts/validate_local.sh
```

Ce script exécute, dans l'ordre demandé : état Git, suite complète (warnings en erreurs), parité/PP, calibration/science/entrypoints, matching, settlement/historique, publication/runtime/pipeline vide, vérification des dépendances et derniers commits. Aucun accès réseau ni secret n'est nécessaire.

La validation locale autorise une revue humaine et la préparation d'un essai isolé. Une fusion en `main` ou une publication de production demande encore la validation externe des sources, une sauvegarde/import de l'historique réel et une feuille Google explicitement de test. Aucune de ces actions distantes n'a été exécutée.
