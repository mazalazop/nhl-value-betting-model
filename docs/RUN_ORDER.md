# Ordre réel d'exécution — POINT

## Chaîne quotidienne

La date passée à `05 --target-date`, `06 --run-date` et `07 --run-date` désigne la **date NHL `gameDate`**, pas la date française du coup d'envoi. Le défaut du workflow est la date à New York. `start_time_utc` conserve l'instant de début du match ; `captured_at` est l'instant UTC de collecte des cotes. Une rencontre de soirée NHL peut tomber le lendemain à Paris.

1. Restaurer `outputs/history/master_daily_bets_history.csv` et `settlement_revisions.csv` depuis la sauvegarde canonique. Le workflow utilise `scripts/restore_history.py`, des artifacts propres à la branche et un token GitHub limité à la lecture. Une initialisation vide exige `bootstrap_history=true` et l'absence d'artifact antérieur ; un artifact expiré ne permet jamais cette initialisation.
2. `python model/00_refresh_sources.py` : calendrier NHL régulier/playoffs, joueurs historiques et roster actuel avec date d'observation. Les boxscores finaux sont partagés via `data/raw/boxscores/`.
3. `python model/00a_refresh_pp_stats.py` : PP des matchs types 2 et 3, plage tirée en priorité de `matchs.csv`, bornée à aujourd'hui. `--force` reste un alias de compatibilité sans effet supplémentaire : ce script recharge déjà toute la plage demandée.
4. `python model/00b_build_base_match_fusionnee.py` : statistiques des matchs explicitement finaux, validation des identités et `points = buts + passes`.
5. `python model/09_settle_previous_bets.py --run-date "$RUN_DATE"` : règlement local, préservation des cas inconnus, reprise des corrections NHL et journal des révisions. Aucune publication directe dans ce script.
6. `python model/00c_refresh_team_standings.py` : snapshots J−1 des dates historiques, plus hier pour le contexte futur. Les consommateurs rejettent les snapshots hors saison, trop anciens ou incohérents avec `api_date`.
7. `python model/01_build_base_features.py` : bases canonique, features et contexte. PP : couverture minimale 90 % globale et par saison/type disponible, blocage par défaut. `--pp-coverage-policy warn` permet un diagnostic contrôlé sans inventer les PP absents.
8. `python model/05_predict_upcoming_games.py --target-date "$RUN_DATE"` : roster frais (48 h), sinon fallback historique explicite de 45 jours ; entraînement/calibration sur les seules dates antérieures ; prédictions et bundle `.joblib` avec manifeste. Un roster actuel ne peut servir à reconstruire rétroactivement l'effectif d'une date passée.
9. Exécuter `scripts/collect_unibet_structured.py`, copier `outputs/structured_bookmaker/normalized_points_odds.json` vers `outputs/normalized_points_odds.json`, puis `scripts/validate_bookmaker_contract.py outputs/normalized_points_odds.json`. Une erreur source/identité empêchant toute collecte bloque ; une vraie absence de marché est distincte.
10. `python model/06_match_model_to_unibet_odds.py --run-date "$RUN_DATE"` : matching strict, calcul local de `1 / odds`, edge, EV et Kelly, exports des rejets. Le flag historique `--disable-fuzzy` est conservé ; le fuzzy approximatif est toujours désactivé.
11. `python model/07_build_daily_bets.py --run-date "$RUN_DATE"` : sélection et ajout canonique sous verrou. Les valeurs initiales d'un pari déjà enregistré restent immuables à la republication.
12. Facultatif, après revue et avec une **feuille de test** : `08_publish_to_google_sheet.py --sheet-id ... --credentials-env`. Les deux vues sont écrites dans une même requête batch ; aucune séquence `clear → write`. L'historique déjà présent dans Sheets doit être couvert par le canonique, sinon la publication bloque.
13. Sauvegarder l'historique, même si une étape ultérieure échoue. Les artifacts GitHub ne remplacent pas une sauvegarde durable indépendante.

La publication est désactivée par défaut et interdite sur `astra/audit-remediation`. Sur cette branche seulement, le premier historique vide peut être initialisé automatiquement en l’absence totale d’artifact antérieur ; les reruns restaurent toujours l’historique. `scripts/validate_sheets_readonly.py` vérifie les métadonnées et en-têtes sans aucune écriture. L’authentification utilise `GOOGLE_CREDENTIALS` uniquement en mémoire. Aucun pari n'est déclenché par ces scripts.

## Contrôle Sheets et migration exceptionnelle

Les pushes et les runs ordinaires ne migrent jamais `history_raw` : tests offline, authentification avec scope readonly et validation des headers seulement. Un schéma canonique reste accepté après ajout de lignes ou modification légitime de résultats ; aucun fingerprint de données historique ne conditionne ce contrôle. Des colonnes manquantes provoquent un refus explicite.

La migration est une opération manuelle exceptionnelle. Simuler d'abord `python scripts/migrate_history_sheet.py` avec authentification en mémoire ; vérifier les headers, le nombre de lignes, les empreintes des cellules projetées et les résultats conservés. Après revue seulement, déclencher le workflow sur la branche de remédiation avec `migrate_history_schema=true` et `migration_expected_snapshot` égal au SHA256 exact du rapport. L'application atomique refuse un snapshot changé et vérifie immédiatement les cellules après écriture. Laisser `publish_to_sheet=false`. Les runs suivants gardent la migration désactivée et vérifient seulement le schéma. Ne jamais remplacer un fingerprint pour contourner une divergence non revue.

Diagnostic du 4 octobre 2026, run #224 : 56 lignes, 18 headers legacy, aucun doublon, trois colonnes manquantes (`id_match`, `id_joueur`, `outcome_key`). Simulation : 56 lignes conservées, toutes les cellules et résultats inchangés, six requêtes d'insertion/headers seulement. Le snapshot `2694bdea9fae83097cc296b113daea9aa1eda73f5345abbcb67c988139ed547f` diffère de celui de la migration initiale : l'application automatique refusait donc correctement de poursuivre. Le diagnostic ne permet pas d'attribuer à lui seul le retour au schéma legacy à un auteur ou un run. Tout autre producteur de cette feuille doit conserver le schéma canonique, sinon le contrôle readonly refusera de nouveau la feuille.

## Évaluation scientifique séparée

```bash
python model/02_train_point_model.py
python model/03_calibrate_point_model.py
python model/10_compare_point_model_to_baseline.py
python model/11_evaluate_point_walk_forward.py
```

- `02` conserve le découpage historique par dates 70/15/15 et entraîne les variantes POINT existantes. Suppression de colonnes constantes/entièrement absentes sur le train seulement ; NaN gérés nativement par HGB.
- `03` apprend/choisit le calibrateur dans la validation, puis mesure le test strictement postérieur. La métrique « validation calibrée » après refit du calibrateur est **in-sample** et signalée comme telle.
- `05` réentraîne sur l'historique disponible avec 85 % des dates pour fit et 15 % pour calibration. Ses fonctions de calibration et paramètres modèle sont communs à l'évaluation, mais sa période de fit n'est pas celle du benchmark fixe `02`.
- `11` évalue précisément cette procédure de fit/calibration en fenêtres croissantes. Les derniers 15 % des dates restent réservés et inutilisés. Sorties : Brier, log-loss, AUC, AP, précision/lift top 10 %, bins de calibration, top 5/10 quotidiens et baseline constante calculée sur le passé.
- `10` distingue les métriques absentes, inchangées, meilleures et moins bonnes. Il ne décide jamais d'un déploiement sur le test final. Sans identité du dataset historique de référence, la comparabilité reste non vérifiée.
- `02b` est une ablation exploratoire distincte avec sa propre population et ses paramètres historiques ; elle ne doit pas être comparée au benchmark principal comme un changement isolé. Son early stopping aléatoire et les labels manquants convertis en zéro ont été supprimés.
- `04` (BUT) est hors chaîne POINT. Il entraîne un modèle de recherche avec `--features`, `--output` et `--phase prepare|baseline|families|pk|models|final`. La phase PK exige `--pk`. Respecter cet ordre; aucun réglage après ouverture du final. Le premier candidat n'a pas franchi le critère de promotion opérationnelle : voir [l'étude BUT](GOAL_SCIENCE_20261004.md).

## Cache et limites

Cache uniquement pour des boxscores d'identité vérifiée et de statut final : durée de vie 1 h pour les matchs des sept derniers jours, 30 jours pour les autres. `00 --full-refresh` et `00b --full-refresh` permettent une vérification intégrale, notamment pour des corrections anciennes. Les statistiques PP restent un refresh complet. Les appels externes et durées sur plusieurs saisons n'ont pas été mesurés ici.

Les bundles `.joblib` sont des artefacts locaux de confiance ; ne pas charger de fichier de provenance inconnue. Les manifestes enregistrent commit, versions, paramètres et empreintes SHA-256 des entrées explicites, sans lire les secrets.
