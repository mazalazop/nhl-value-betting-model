# NHL Value Betting Model

Repo dédié à la partie modèle du projet NHL value betting.

## Priorités
1. Build base features
2. Train point model
3. Calibrate point model
4. Train goal model

## Structure
- `model/` : scripts principaux
- `configs/` : fichiers de configuration
- `notebooks/` : notebook de pilotage
- `outputs/` : résultats
- `docs/` : documentation

## Règle
Le notebook Colab actuel reste inchangé pour l’instant.
La structuration se fera progressivement, sans casser l’existant.

## Règle projet — ajout de features

Avant d’ajouter de nouvelles features :
- lister à l’utilisateur tous les critères déjà utilisés par le modèle
- expliquer simplement le fonctionnement du modèle
- préciser ce qui est volontairement exclu
- demander à l’utilisateur s’il souhaite ajouter de nouvelles features

Important :
- un seul bloc de nouvelles features à la fois
- validation temporelle obligatoire après chaque ajout
- aucune feature utilisant une information connue après le début du match
