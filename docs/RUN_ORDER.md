# RUN ORDER — NHL Value Betting Model

## Principe général
Le repo GitHub est la source officielle du projet.

Objectif :
1. reconstruire les sources amont propres
2. reconstruire la base match fusionnée
3. construire les features
4. entraîner le modèle POINT
5. calibrer le modèle POINT
6. entraîner le modèle BUT
7. intégrer plus tard les cotes Unibet
8. comparer probabilité modèle vs probabilité implicite
9. calculer l’edge
10. traiter plus tard la bankroll

---

## Ordre d’exécution

### 0. Rafraîchir les sources amont
Script :
- `model/00_refresh_sources.py`

Rôle :
- reconstruit `data/raw/joueurs.csv`
- reconstruit `data/raw/matchs.csv`

Sorties attendues :
- `data/raw/joueurs.csv`
- `data/raw/matchs.csv`

Condition pour continuer :
- les 2 fichiers existent
- les colonnes sont correctes
- pas de dépendance Supabase

---

### 0b. Construire la base match fusionnée
Script :
- `model/00b_build_base_match_fusionnee.py` *(à créer)*

Rôle :
- reconstruire les sources intermédiaires nécessaires
- produire `base_match_fusionnee.csv`

Entrées attendues :
- `data/raw/joueurs.csv`
- `data/raw/matchs.csv`
- source stats reconstruite proprement

Sortie attendue :
- `data/raw/base_match_fusionnee.csv`

Condition pour continuer :
- `base_match_fusionnee.csv` existe
- pas de match manquant
- pas d’incohérence équipe / adversaire
- structure cohérente avec l’ancien pipeline validé

---

### 1. Construire les bases features
Script :
- `model/01_build_base_features.py`

Entrée attendue :
- `data/raw/base_match_fusionnee.csv`

Sorties attendues :
- `data/final/base_canonique_v2.csv`
- `data/final/base_features_v2.csv`
- `data/final/base_features_context_v2.csv`

Condition pour continuer :
- les 3 fichiers sont créés
- pas d’erreur bloquante
- contrôles temporels conservés

---

### 2. Entraîner le modèle POINT
Script :
- `model/02_train_point_model.py`

Entrée attendue :
- `data/final/base_features_context_v2.csv` *(ou autre fichier final validé selon version retenue)*

Sortie attendue :
- modèle POINT entraîné
- prédictions out-of-fold / validation temporelle
- métriques de performance

Condition pour continuer :
- validation temporelle en place
- aucune fuite de données
- qualité modèle jugée suffisante

---

### 3. Calibrer le modèle POINT
Script :
- `model/03_calibrate_point_model.py`

Rôle :
- calibrer les probabilités du modèle POINT

Sortie attendue :
- probabilités POINT calibrées
- diagnostic de calibration

Condition pour continuer :
- calibration meilleure ou au minimum propre et documentée

---

### 4. Entraîner le modèle BUT
Script :
- `model/04_train_goal_model.py`

Rôle :
- entraîner le modèle joueur 1+ but

Condition :
- ne commencer qu’après validation du pipeline POINT

---

## Priorité métier
Ordre de priorité :
1. données propres
2. modèle POINT
3. calibration POINT
4. modèle BUT
5. cotes Unibet
6. edge
7. bankroll plus tard

---

## Ce qu’on ne fait pas
- ne pas repartir sur le scraper
- ne pas dépendre de Supabase pour `joueurs.csv` et `matchs.csv`
- ne pas contourner un contrôle bloquant
- ne pas avancer au modèle si la base amont n’est pas propre

---

## État actuel du projet
Validé :
- scraper Unibet séparé
- `model/01_build_base_features.py` déjà rempli
- `model/00_refresh_sources.py` ajouté
- repo GitHub = source officielle

En cours :
- création de `model/00b_build_base_match_fusionnee.py`

À venir :
- `02_train_point_model.py`
- `03_calibrate_point_model.py`
- `04_train_goal_model.py`
