# État initial — remédiation locale

Commit audité et point de départ : `c51d544b9b7d7bf49d6b3c39740fffdb8ab35565`.
Working tree propre avant création de `astra/audit-remediation`.
Python disponible : 3.11.9. Aucune installation initiale de pandas, numpy,
scikit-learn ou gspread dans cet interpréteur. Un venv isolé est utilisé pour
tester les versions retenues ; aucun credential n'est consulté.

## Données et référence

15 scripts Python, README, RUN_ORDER et un workflow GitHub Actions.
Les dossiers data, outputs, configs et notebooks ne contiennent que .gitkeep.
Aucun modèle sérialisé, calibrateur, CSV de données, prédiction, métrique
d'exécution ou historique de paris n'est disponible au départ.
La référence enrichie sigmoid (Brier 0.205518, log-loss 0.599639,
AUC 0.678702) existe uniquement comme constantes dans le script 10.
Elle est documentaire, NON reproduite par cette mission.

## Architecture et paramètres initiaux

Évaluation : 00 / 00a / 00b / [00c optionnel] -> 01 -> 02 -> 03 -> 10.
Production : scraper externe -> 00 -> 00a -> 00b -> 01 -> 05 -> 06 -> 07 -> 08.
09 n'est pas connecté. 04 est un placeholder et reste hors périmètre.
02b réalise une ablation distincte de la production.

POINT : 60 features baseline + 21 de contexte, whitelist identique entre 02/05.
HistGradientBoostingClassifier : learning_rate=.05, max_iter=300, max_depth=6,
min_samples_leaf=50, l2_regularization=1, early_stopping=False, random_state=42.
Pondération positives = négatives / positives dans le train.
02 : dates uniques 70/15/15. 05 : dates historiques 85/15 fit/calibration.
03 : logistic(logit(p)) ; 05 : logistic(p), donc divergence.
07 : 10 picks, cote minimale 1.40 ou p>=.90, value indicative >=.02,
un pick par nom de joueur ; EV négative autorisée.

## Risques prioritaires à caractériser

Parité saison équipe, defaults et NaN ; calibration divergente ; absence de
finalisation ; PP inconnu remplacé par zéro ; standings sans saison/fraîcheur ;
matching sans date/marché ; perte du drapeau streak ; historique non restauré ;
settlement absent ; clear/write Sheets ; inputs shell interpolés ; versions libres.

La suite utilise des fixtures synthétiques et des mocks réseau/Sheets.
Les résultats synthétiques ne représentent ni performance NHL ni gain prédictif.
