# RUN ORDER

## Objectif
Ce repo sert à structurer progressivement la partie modèle du projet NHL value betting, sans casser le notebook Colab historique.

Le notebook Colab actuel reste la référence source tant que les scripts propres ne sont pas reconstruits et validés.

---

## Ordre officiel de travail

### 1. Build base features
Fichier :
- `model/01_build_base_features.py`

But :
- reconstruire les bases propres du projet à partir des sources validées
- produire :
  - `data/final/base_canonique_v2.csv`
  - `data/final/base_features_v2.csv`
  - `data/final/base_features_context_v2.csv`

---

### 2. Train point model
Fichier :
- `model/02_train_point_model.py`

But :
- entraîner le modèle POINT
- produire les métriques et prédictions historiques du modèle POINT

---

### 3. Calibrate point model
Fichier :
- `model/03_calibrate_point_model.py`

But :
- calibrer les probabilités du modèle POINT
- produire les prédictions calibrées

---

### 4. Train goal model
Fichier :
- `model/04_train_goal_model.py`

But :
- entraîner le modèle BUT
- produire les métriques et prédictions historiques du modèle BUT

---

## Ce qu’on ne fait pas encore
Les étapes suivantes viendront plus tard :

- prédictions sur matchs à venir
- fusion modèle + cotes Unibet
- calcul de l’edge
- export Google Sheets
- bankroll management

---

## Règles importantes
- ne pas modifier le notebook Colab historique pour l’instant
- reconstruire progressivement la logique dans des scripts propres
- avancer bloc par bloc
- ne pas mélanger modèle, edge et bankroll trop tôt
- priorité : POINT, puis calibration POINT, puis BUT

---

## Référence actuelle
Le notebook Colab historique reste la source de vérité actuelle pour :
- la logique data
- la logique modèle
- les sorties déjà validées

Les scripts du repo modèle sont pour l’instant des placeholders propres, destinés à être remplis progressivement.
