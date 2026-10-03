# Migration sûre du schéma history_raw

Le contrôle réel du run `37130070589` a confirmé 18 colonnes, sans extra, dans cet ordre :

```
bet_id, run_date, date_match, player_name, team, opponent, bookmaker, market,
stat, threshold, odds_decimal, implied_probability_pct, model_probability_pct,
edge_probability_pct, bet_status, result, actual_stat_value, settled_at
```

Le run `37130980452` a ensuite simulé la migration sur les **46 lignes réelles**, avec
authentification en mémoire et scope readonly. Aucun identifiant, résultat, statut,
valeur réelle ou timestamp de règlement n'a changé. Aucun doublon existant ou ajouté.
Les empreintes de toutes les cellules originales et de leur projection après
simulation sont identiques :
`8eaedc88afecb739a34f913e23a0654451cd1575b9d86991c7669a3d21f0e137`.

## Opération autorisée

`scripts/migrate_history_sheet.py` prend le schéma directement dans
`HISTORY_OUTPUT_COLUMNS` de 08. Pour ce cas réel, il insère trois colonnes à gauche
et écrit seulement leurs en-têtes : `id_match`, `id_joueur`, `outcome_key`.
Les cellules historiques de ces nouvelles colonnes restent vides, faute de preuve
permettant de les reconstruire. Les anciennes colonnes sont déplacées par Google
Sheets, pas réécrites depuis un DataFrame ou des valeurs formatées.

Une seule requête `batch_update` contient les six opérations. Aucun `clear`, aucune
suppression de ligne/colonne, aucune réécriture de cote, probabilité ou résultat.
`daily_picks` n'est pas modifié. Le plan générique conserve les colonnes legacy
supplémentaires et ne déduplique pas arbitrairement des données existantes.

Le snapshot est lu sous forme de `userEnteredValue` typées : textes, nombres et
booléens sont comparés sans conversion. Une migration structurelle comportant des
formules, des en-têtes dupliqués ou des cellules sans en-tête est refusée pour revue.
La simulation compare chaque cellule historique, pas seulement les lignes réglées.

## Garde-fous et relance

L'écriture exige l'empreinte exacte du snapshot simulé et revu :
`20fe57c86389392af71c0d01cdcd459c2d699e6103648794ee0eea14317bae64`.
Une seconde lecture immédiate détecte un changement concurrent avant le batch.
Après écriture, toutes les cellules sont relues et comparées au résultat prévu.
Un second passage vérifie l'idempotence : zéro requête d'écriture supplémentaire.
Un schéma déjà conforme ne doit pas satisfaire l'ancienne empreinte puisqu'il
n'entraîne aucune modification ; toutes ses valeurs restent intactes.

Sheets ne propose pas ici de comparaison-et-écriture conditionnelle atomique :
les runs GitHub sont sérialisés et une double lecture réduit le risque d'édition
manuelle concurrente ; la vérification après écriture reste obligatoire.
Un échec de vérification bloque, sans tentative de réécriture aveugle.

Le workflow de remédiation conserve les rapports de l'application, du rerun et du
contrôle readonly dans l'artifact `henachel-sheets-readonly-<run_id>`. Aucun contenu
de credentials ni aucune ligne de paris n'est imprimé ou exporté par ces rapports.
La publication métier de 08 reste désactivée sur cette branche.

## Tests

```bash
.venv/bin/python -m pytest tests/test_sheets_migration.py tests/test_sheets_environment.py tests/test_publication.py tests/test_history.py tests/test_settlement.py -q -W error
bash scripts/validate_local.sh
```

Les tests couvrent le schéma de production exact, les résultats win/loss/void,
les identifiants connus, les valeurs inconnues vides, une feuille vide, des extras,
le rerun, les modifications concurrentes et une erreur API sans effet destructif.
Cette migration de schéma ne prétend pas réconcilier les anciennes identités avec
l'historique CSV canonique : aucune identité NHL n'est inventée.
