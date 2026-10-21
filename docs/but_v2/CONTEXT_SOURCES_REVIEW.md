# Enrichissement pré-match — faisabilité, sans changement du modèle

Pages publiques consultées le 20 octobre 2026; revue du code local reprise le 21.
Aucun nouveau champ ajouté aux features BUT V2. Les sources restent à qualifier
sur des captures prospectives; pas de reconstruction historique des confirmations.

| Donnée | Source examinée | Preuve et manque | Traitement proposé |
|---|---|---|---|
| Gardien projeté/confirmé | [Daily Faceoff](https://www.dailyfaceoff.com/starting-goalies) | La page montre rencontres, heure UTC, statut et, pour certaines confirmations, heure et lien vers une source journalistique. Les entrées non confirmées peuvent ne pas être horodatées. | Archiver HTML/JSON brut, hash, collecte UTC et horodatage propre à la confirmation; résoudre ID NHL par équipe/roster. Unconfirmed ne signifie pas confirmé ni nécessairement probable. |
| PP1/PP2 | [RotoWire lineups](https://www.rotowire.com/hockey/nhl-lineups.php) | Unités PP affichées; le parseur existant n'a pas démontré d'horodatage de mise à jour propre à l'unité. | Conserver observation brute et dernier état connu, âge et UNKNOWN si fraîcheur non prouvée. Ne jamais utiliser l'heure d'une confirmation gardien comme heure PP1. |
| Temps PP historique | API officielle NHL, `api.nhle.com/stats/rest/en/skater/powerplay` et boxscores `api-web.nhle.com/v1/gamecenter/{id}/boxscore` | Le collecteur local `context_sources.py` vérifie pagination/identités et garde les matchs contributeurs. PP TOI n'est pas toujours présent dans les boxscores. | Sommes/moyennes seulement de rencontres finales observées avant émission; zéro uniquement si explicitement fourni. Ce n'est pas PP1 confirmé. |
| xG individuel | [MoneyPuck — téléchargements](https://moneypuck.com/data.htm) et [méthode](https://www.moneypuck.com/about.htm) | Données de tirs avec scores xGoals; tirs bloqués exclus du jeu de tirs décrit. Modèle tiers, pas xG officiel NHL. Les corrections historiques ne prouvent pas l'état disponible à une date passée. | Télécharger uniquement les fichiers publiés autorisés, archiver la version avant match, joueur/match/situation, dernière rencontre utilisée, hash et version du modèle si disponible. Aucune distance de tir rebaptisée xG. |

MoneyPuck indique un usage gratuit non commercial et des restrictions de scraping;
crédit obligatoire, autres usages à clarifier avec le fournisseur avant automatisation.
Aucun abonnement souscrit, aucun accès payant contourné, aucun message envoyé.
Les pages publiques ne garantissent pas un contrat API stable ni une couverture complète.

Le code existant `henachel/context.py` sait conserver CONFIRMED/PROBABLE/UNKNOWN,
source_timestamp, collected_at_utc, âge et TTL; il ramène une information non
horodatée ou périmée à UNKNOWN. Réutiliser le schéma dans un **registre contextuel
shadow distinct**, sans écrire dans le SQLite POINT/V1 existant.

Pour chaque observation future : game_id, player_id (ou goalie_id), équipe,
source URL, contenu brut/hash, source_timestamp nullable, collected_at UTC,
publication_state, confidence, source_age, TTL, dernière date de données,
raison d'absence. Une projection doit rester une projection. En cas de conflit,
conserver les sources concurrentes; ne pas choisir rétrospectivement celle qui
correspond au gardien réellement aligné.

Qualification progressive : collecte seule, puis contrôle de couverture et
horaires pendant 30 jours, audit ID/équipe et valeurs manquantes, protocole
séparé d'ablation hors échantillon. La configuration BUT V2 actuelle reste figée.
Sources sans disponibilité historique prouvée : usage futur seulement, jamais
backfill du replay en prétendant connaître les confirmations avant match.
