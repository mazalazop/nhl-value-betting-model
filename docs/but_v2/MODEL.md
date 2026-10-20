# Modèle indépendant BUT V2

`henachel/but_v2.py`, configuration `configs/but_v2.json`.
Aucun import depuis les pipelines existants. Candidat shadow logistique fixé
avant évaluation; Poisson et HGB sont comparateurs, sans sélection automatique.
Régression logistique L2 C=1, Poisson alpha=1 avec P(>=1)=1-exp(-lambda),
HGB 150 itérations/15 feuilles. Pas de pondération de classes dans V2.
La loi binomiale négative est différée : le score cible est binaire et la
surdispersion ne justifie pas à elle seule une complexité supplémentaire.

Features reconstruites des mesures brutes : tirs/60 et moyennes 3/5/10,
TOI et PP TOI historiques 3/5/10 (proxys de projection, jamais PP1), buts,
conversion, fréquence joueur, repos/domicile, disette et série de buts,
contexte buts équipe et encaissés adversaire sur 5 rencontres antérieures.
Les fenêtres se réinitialisent par saison; aucun bonus/malus de série.
Un joueur avec plusieurs rencontres au même jour est refusé : ordre inconnu.
Les valeurs inconnues restent NaN dans les features; médiane et indicateur
de manque ajustés sur le train uniquement dans l'estimateur.

15% des dernières dates d'entraînement servent à la calibration temporelle
existante (raw/sigmoid/isotonic); aucune date test pour choisir le calibrateur.
La calibration est évaluée hors échantillon, pas présumée efficace.

Les mesures brutes sont des statistiques finales téléchargées ultérieurement :
la reconstruction n'est pas une archive pré-match. Pour le shadow, n'accepter
que l'historique effectivement capturé avant émission et les matchs futurs.
Aucune donnée actuelle de buts/tirs/TOI n'est admissible dans les lignes futures.
Les futures compositions doivent être horodatées et sourcées avant ajout.

Tests : invariance par préfixe, corruption des résultats futurs sans effet
sur fit/calibration, whitelist, identités, inconnus et remise à zéro saison.
Retour arrière : retirer le module, ses tests et sa configuration; aucun
poids POINT, modèle BUT V1 ni archive officielle à restaurer.
