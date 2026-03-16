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

## Règle de pilotage métier du modèle

Avant toute décision d’ajout de nouvelles features, il faut :

1. présenter à l’utilisateur la liste claire des critères actuellement pris en compte par le modèle
2. expliquer simplement comment le modèle fonctionne
3. distinguer :
   - les variables de forme récente
   - les variables de contexte match
   - les variables liées à l’adversaire
   - les variables historiques
4. signaler explicitement les variables exclues pour éviter toute fuite de données
5. demander ensuite à l’utilisateur s’il souhaite ajouter de nouvelles features

Règle :
- ne pas ajouter de nouvelles features sans passage préalable par cette étape d’explication
- procéder ensuite par ajout d’une seule famille de features à la fois
- revalider après chaque ajout avec validation temporelle
