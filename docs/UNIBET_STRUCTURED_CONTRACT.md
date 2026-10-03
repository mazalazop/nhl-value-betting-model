# Contrat Unibet structuré

Le collecteur `scripts/collect_unibet_structured.py` lit le hub NHL public et le
bloc `EventsDetail.events` du JSON SSR `serverApp-state` des pages événement.
Il ne conserve que les événements, jamais les champs de session de la page.
Il ne dépend plus du scraper Playwright externe ni du runner macOS.

Chaque réponse est horodatée à sa réception (`captured_at`, également exposé sous
`collected_at_utc`). Un ancien document n'est jamais réhorodaté. `parsedStart`
fournit le début UTC bookmaker ; la durée maximale de fraîcheur reste six heures.

Une jointure calendrier NHL exige exactement un match futur avec les deux équipes
et la même heure UTC. La date `gameDate` et l'orientation domicile/extérieur viennent
de ce match NHL vérifié, jamais de l'ordre visuel bookmaker. Les alias sont explicites.
Le joueur doit correspondre à un nom normalisé exact et unique dans les rosters
des deux équipes, observés depuis moins de 24 heures. Les accents et la ponctuation
sont normalisés ; les initiales non vérifiables, homonymes et contradictions de
roster sont rejetés. Un roster n'est pas une garantie de participation au match.

Seul le groupe observé `4267`, libellé `Nombre de Points - Joueur - Match (Hors TAB)`,
avec période `Match`, est accepté. Une unique issue `<joueur> 1+`, visible et non
suspendue, doit appartenir au bon événement et au bon marché. Aucun 2+/3+/4+ n'est
converti en 1+. Les odds sont décimales, finies et supérieures à 1.

L'enveloppe compatible avec 06 contient `bookmaker`, `market`, `stat`, `threshold`,
`outcome_label`, `rows`. Chaque ligne conserve notamment :

- `player_name`, `team`, `home_team`, `away_team` ;
- `event_id`, `event_url`, `event_slug`, `event_start_utc` ;
- `date_match` NHL, `nhl_game_id`, `nhl_player_id` ;
- `captured_at`, `collected_at_utc`, `roster_observed_at` ;
- marché POINT 1+, cote et probabilité implicite recalculée ;
- IDs marché/issue et description de la preuve d'identité.

Le matching de 06 reste inchangé et refait ses contrôles. Le validateur
`scripts/validate_bookmaker_contract.py` refuse métadonnées manquantes, NaN,
timestamps invalides, cotes obsolètes, événements contradictoires et sélections
dupliquées. Les rapports distinguent les événements rejetés, les marchés joueur
rejetés et les lignes soumises à 06. Une collecte vide après des erreurs techniques
ne constitue pas une journée sans marché valide.

## Essai réel du 3 octobre 2026

La première collecte compatible a produit 116 lignes vérifiées sur 12 matchs,
116 rapprochements, zéro rejet dans 06 et 10 picks dans un historique isolé.
Les 564 prédictions et données NHL déjà validées ont été réutilisées. La publication
a été simulée avec deux appels mock, sans accès de production. Un alias supplémentaire
réellement observé, `TOR MapleLeafs`, a ensuite été ajouté pour couvrir Toronto.
Les prix évoluent ; les comptes finaux sont consignés dans le rapport de session.

`tests/data/unibet_real_2026_10_03.json` est un extrait réellement capturé (un marché,
ses issues, le match et les rosters associés). Son prix 1+ était 2,00 à la collecte.
Les cas d'erreur des tests sont des mutations explicitement synthétiques de cet extrait.

## Exécution distante sûre

Sur `astra/audit-remediation`, un push déclenche le workflow de cette branche.
Il utilise le collecteur structuré après le refresh NHL et les prédictions. Le
premier historique est vide et propre à cette branche ; aucune copie de paris de
production n'est utilisée. Les runs suivants restaurent cet historique.

La publication métier est interdite sur cette branche. Le job Sheets utilise
uniquement le scope `spreadsheets.readonly`, lit le titre et les en-têtes de
`daily_picks`/`history_raw`, et publie un rapport sans données de paris ni secrets.
L'authentification service-account se fait en mémoire via `GOOGLE_CREDENTIALS`.
Aucun fichier de credentials n'est créé. Les permissions d'écriture et un
append/update réel ne sont pas validés en l'absence de sandbox dédiée.

```bash
.venv/bin/python scripts/collect_unibet_structured.py --raw-dir /chemin/isole/data/raw
.venv/bin/python scripts/validate_bookmaker_contract.py outputs/structured_bookmaker/normalized_points_odds.json
bash scripts/validate_local.sh
```

Aucune optimisation du modèle, feature, calibration ou règle de sélection n'est incluse.
