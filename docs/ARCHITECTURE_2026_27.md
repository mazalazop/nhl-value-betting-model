# Architecture cible NHL Value Betting 2026-27

## Objectif

Transformer le pipeline historique en moteur quotidien stable, puis exposer le résultat à deux clients mobiles : iOS et Android.

NHL / données joueur + matchs
-> ingestion / contrôles qualité
-> features temporelles
-> modèle POINT + calibration
-> matching des cotes
-> Pick Engine
-> payload JSON / API
-> iOS + Android

## Convention de slate France

Le projet utilise Europe/Paris comme référence métier.

Un match NHL joué mardi soir en France et terminé mercredi matin conserve le slate du mardi en France. Cette convention doit être utilisée pour les prédictions, les cotes, l'historique, le settlement, l'API et l'affichage mobile.

La source de données doit conserver séparément la date officielle NHL et la date métier française.

## Séparation des responsabilités

- data/raw : données sources.
- data/final : datasets dérivés et features.
- outputs : artefacts d'un run.
- model : logique scientifique et métier.
- schemas : contrats d'échange.
- configs : paramètres explicites.
- outputs/api : publication consommable par les clients.
- ios et android : clients, sans logique de calcul de probabilité.

## Règles

1. Aucun signal connu après le début du match dans les features pré-match.
2. Validation temporelle obligatoire.
3. Les probabilités publiques viennent de la version calibrée retenue.
4. Le moteur de sélection est déterministe à entrée identique.
5. Une donnée absente ne doit pas devenir silencieusement un signal positif.
6. Une cote absente n'est jamais inventée.

## Publication

Le pipeline produit d'abord outputs/07_daily_bets.csv, puis outputs/api/daily_picks.json.

Google Sheets reste une sortie de transition pour audit humain. Les applications mobiles ne dépendent pas de Google Sheets.

## Mobile

iOS et Android partagent le même contrat API. Les clients ne calculent pas les picks et ne stockent aucun secret du pipeline.

## Livraison

1. Stabiliser le pipeline 2026-27.
2. Mettre en place le contrat JSON.
3. Ajouter validation automatique et tests.
4. Mettre en place le backtest walk-forward.
5. Déployer une API de lecture.
6. Construire iOS et Android.
7. Ajouter notifications, historique et métriques.
8. Préparer conformité App Store / Google Play et politique de confidentialité.
