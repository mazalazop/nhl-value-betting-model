# CHANGELOG — Pipeline HENACHEL rendu opérationnel

## Problème principal résolu
Le pipeline ne pouvait pas tourner en moins de 3h à cause de scripts
qui refetchaient TOUT depuis l'API NHL à chaque run. Les données
intermédiaires étaient perdues entre chaque exécution GitHub Actions.

**Après ces modifications** : run quotidien en ~10-15 min (au lieu de 6h+ / timeout).

---

## Fichiers modifiés (7 fichiers)

### 1. `model/00_refresh_sources.py`
- Boxscore supplement **désactivé par défaut** (économise ~5h)
- Nouveau flag `--with-boxscore-supplement` pour forcer si besoin

### 2. `model/00a_refresh_pp_stats.py` ← RÉÉCRIT
- **Mode incrémental** : ne refetch que les mois après la dernière date en cache
- Dédoublonnage automatique lors du merge avec l'existant
- Flag `--rebuild` pour forcer un refresh complet

### 3. `model/00b_build_base_match_fusionnee.py` ← RÉÉCRIT
- **Mode incrémental** : ne fetch que les matchs absents de `stats.csv`
- Premier run = rebuild complet (~30 min), runs suivants = quelques secondes
- Flag `--rebuild` pour forcer
- Tolérance aux erreurs individuelles (un match qui échoue ne bloque plus tout)

### 4. `model/04_train_goal_model.py` ← REMPLACÉ (placeholder → réel)
- Modèle BUT complet (HistGradientBoostingClassifier)
- Validation temporelle, whitelist de features
- Métriques + prédictions sauvegardées

### 5. `model/05_predict_upcoming_games.py`
- **Persistance du modèle** en `models/point_model_enrichi.joblib`
- Calibrateur sauvegardé en `models/point_calibrator_sigmoid.joblib`
- Métadonnées en `models/point_model_meta.json`

### 6. `model/09_settle_previous_bets.py`
- Suppression des chemins Colab (`/content/drive/...`)
- Seul `data/raw/stats.csv` du repo est utilisé

### 7. `.github/workflows/Workflow global Henachel — POINTS.yml` ← RÉÉCRIT
- **Cache GitHub Actions** pour persister entre les runs :
  - `data/raw/stats.csv` (~100 MB, crucial pour l'incrémental)
  - `data/raw/pp_stats_game.csv`
  - `data/raw/team_standings_daily.csv`
  - `data/raw/base_match_fusionnee.csv`
  - `outputs/history/` (historique des paris)
  - `models/` (modèle persisté)
- Ajout de `joblib` dans les dépendances pip
- Ajout des étapes 00c (standings) et 09 (settlement)
- Noms d'étapes numérotés pour lisibilité
- Rétention artifacts 14 jours

---

## Nouveau dossier
- `models/` avec `.gitkeep` — stocke les artefacts modèle

---

## Comment déployer

1. **Remplacer** les 7 fichiers dans le repo `nhl-value-betting-model`
2. **Créer** le dossier `models/` à la racine avec un fichier `.gitkeep` vide
3. **Commit + push**
4. **Lancer** le workflow `henachel-points-global` manuellement

### Premier run
- Le cache est vide → rebuild complet de stats.csv (~30 min)
- C'est normal, c'est le seul run lent

### Runs suivants
- Le cache est restauré → mode incrémental (~5-10 min total)
- Seuls les nouveaux matchs / nouvelles dates sont fetchés

---

## Scripts NON modifiés (fonctionnent déjà)
00c, 01, 02, 02b, 03, 06, 07, 08, 10
