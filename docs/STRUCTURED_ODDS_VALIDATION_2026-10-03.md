# Résultat opérationnel : ingestion structurée Unibet

## Résultat local réel

Collecte du 3 octobre 2026, de `13:43:04.654846Z` à `13:43:13.269356Z` :

- 17 pages événement Unibet inspectées ;
- 127 cotes POINT 1+ normalisées et valides ;
- 21 marchés joueur écartés en amont : absence d'une issue 1+ unique, visible,
  non suspendue et portant le nom vérifiable attendu ; ces marchés ne sont pas
  comptés comme des cotes valides puis rejetées par 06 ;
- zéro événement rejeté pour identité après ajout de l'alias observé `TOR MapleLeafs` ;
- 127 cotes acceptées par le matching existant, zéro rejet dans 06 ;
- 127 joueurs rapprochés sur les 13 matchs NHL prévus ;
- 564 prédictions préexistantes réutilisées, aucun nouvel entraînement/tuning ;
- 10 picks simulés ; 10 identités uniques dans le ledger isolé, aucune addition
  lors de la réexécution de 07 ;
- publication simulée de deux vues : 10 lignes quotidiennes et 10 historiques,
  zéro appel réseau Google.

Les noms abrégés ou divergents entre le libellé du marché et celui de l'issue
peuvent être refusés. Cette restriction de couverture est explicite ; aucune
identité incertaine n'a été forcée pour augmenter le nombre de lignes acceptées.

Les preuves locales sont dans `outputs/structured_bookmaker/` : événements publics
sélectionnés, JSON normalisé et `dry_run_summary.json`. Les prédictions, ledger et
sorties 06/07 sont dans `/private/tmp/henachel-real-validation-n0112i26/outputs/`.
Ces données temporaires ne constituent pas un historique de production.

## Validation locale

`bash scripts/validate_local.sh` : **176 tests passants**, warnings comme erreurs.
Les sous-suites existantes sont également relancées par ce script. Les tests
spécifiques ingestion/contrat, matching, runtime et publication ont été exécutés.
`git diff --check` est sans erreur. Aucun algorithme, feature, calibrateur ou seuil
de sélection n'a été modifié.

## Blocage distant démontré

Le push autorisé a été tenté, puis réessayé avec accès réseau hors sandbox :

```text
fatal: could not read Username for 'https://github.com': terminal prompts disabled
```

Git n'a donc pas d'authentification HTTPS utilisable localement. La recherche via
le connecteur GitHub confirme que la branche distante `astra/audit-remediation`
n'existe pas encore. Aucun workflow de cette branche n'a pu être exécuté.
Le connecteur peut créer des commits distants, mais ne peut pas pousser les objets
Git locaux avec leurs auteurs, dates et identifiants d'origine ; il n'a pas été
utilisé pour remplacer ou condenser l'historique local.

Action externe nécessaire : authentifier Git pour ce repository, sans transmettre
de token à l'assistant. Ensuite :

```bash
GIT_TERMINAL_PROMPT=0 git push origin HEAD:refs/heads/astra/audit-remediation
```

Le push déclenchera le workflow de la branche, incluant le nouveau collecteur et
le contrôle Sheets strictement en lecture seule. Aucun merge dans `main` n'est requis.

## Google Sheets

`scripts/validate_sheets_readonly.py` a été tenté localement : `unavailable`.
Les variables `GOOGLE_CREDENTIALS` et `SHEET_ID` sont absentes (seule leur présence
a été contrôlée, aucune valeur de secret lue ou affichée). Le secret annoncé côté
GitHub n'est pas récupérable pour une exécution locale et n'a pas été copié.

Le job distant est prêt : authentification en mémoire, scope readonly, vérification
du titre Henachel et des colonnes distinctes des vues daily/history, sans écrire
dans les onglets. Il échoue explicitement si le schéma est absent/incompatible.
Il reste à exécuter après déblocage du push. Aucun test d'écriture n'est autorisé
sur la feuille de production sans sandbox dédiée.

## Décision

Le blocage bookmaker local est levé. Les validations Actions et Sheets restent
bloquées par l'authentification Git nécessaire au premier push. **Pas encore prêt
à merger ni à commencer l'optimisation prédictive.** Les fichiers Finder
`.DS_Store` apparus pendant la reprise sont laissés intacts et non committés.
