# Protocole de validation fixé avant calcul

Commande :
```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/evaluate_but_v2.py \
  --features data/final/base_features_context_v2.csv \
  --output outputs/but_v2/replay_20261020
```

Dossier neuf obligatoire. Copie du snapshot, hash source, configuration et
hashes du code enregistrés avant fit. Minimum 160 dates train, fenêtres test
60 dates NHL. Réentraînement/calibration avant chaque fenêtre; aucune
observation du test pour ajuster ce modèle. Les features de chaque match
peuvent utiliser les résultats des jours précédents dans la fenêtre, comme
en opération. V1 est l'algorithme GOAL17 opérationnel avec ses 17 features,
réentraîné au début des mêmes fenêtres, pas ses anciennes probabilités gelées.
Cette cadence de refit n'est pas le refit quotidien de production.

V2 compare logistique, Poisson, HGB. Candidat shadow logistique fixé sans
sélection de gagnant. Ablations logistiques : chaque famille retirée du modèle
complet; fenêtres tirs 3, 5, 10 isolées. Pour une ablation, delta positif signifie
que retirer la famille dégrade; delta négatif qu'il améliore. Les contributions
se chevauchent (TOI entre dans tirs/60); il ne s'agit pas d'effets causaux.
Les familles non mesurées ne sont pas déclarées inutiles.

Toutes les comparaisons exigent exactement les mêmes identités et labels.
Brier, log-loss, calibration par intervalles, Top5/Top10 quotidiens, precision
et recall au seuil descriptif .5, couverture, saison régulière/playoffs,
domicile/extérieur et folds. Bootstrap apparié de blocs circulaires de 7 dates
NHL, 2000 réplications, IC95% et correction Bonferroni sur les variantes.
Le bootstrap n'élimine pas toutes les dépendances répétées joueur/équipe.
Les Top K sont parmi les participants historiques, pas un univers pré-match
prouvant les scratches; leur couverture n'est pas celle des rosters complets.

Pas de ROI historique : aucune cote pré-match vérifiée dans le snapshot.
Pas de test prospectif : dates déjà étudiées, données officielles pouvant
avoir subi corrections tardives. Aucun réglage après lecture du rapport;
aucune promotion automatique, même si une différence est significative.
Validation indépendante requise sur captures CANONICAL futures.

Retour arrière : retirer ces outils et rapports; production indépendante.
