# Autorisation et préinscription A01 — 25 octobre 2026

L'utilisateur a explicitement autorisé dans la conversation ASTRA du 25 octobre
2026 le démarrage prospectif A01 : préinscription, configuration d'activation,
nouveau registre et vérifications du premier fonctionnement réel. Il a interdit
toute modification du modèle et toute promotion de V2. Il exige que le dernier
commit touchant start.json et la variable d'activation précèdent T0, sans antidatage.
Cette note consigne cette autorisation ; elle ne prétend pas être une approbation
déduite d'une CI ou un nouveau consentement pour des corrections officielles.

## Enregistrement retenu

- Étude : **A01-20261026-01**.
- Version : A01_RELEASE_1, 20 fichiers vérifiés ; empreinte canonique :
  `b4200f6577cce493267ebdf33487550a3ed1b550ccdc08844d43d47ff4e50f6b`.
- T0 prévu : **2026-10-25 10:00:00 UTC**, **11:00:00 Europe/Paris (CET)**.
- Première journée NHL admissible : **2026-10-26**.
- Exclusion inclusive : **2026-10-25**.
- Capture admissible : générée après T0, avant tous ses matchs, CANONICAL shadow,
  jour >=2026-10-26 ; mêmes identités et rangs, première capture conservée.
- Registre durable : `henachel-but-v2-a01-A01-20261026-01-`.
- `enabled=true` est inclus dans cette préinscription, pas ajouté après T0.
- Variable HENACHEL_BUT_V2_A01_ENABLED à passer à true avant T0, une fois la fusion
  et les empreintes vérifiées. Le moteur refuse l'exécution avant T0.

## Inventaire antérieur

Les captures examinées sont celles des 22, 23 et 24 octobre 2026 : 539, 86 et
548 paires, soit 1173 au total. Elles restent exploratoires pour A01.
La journée du 25 est également exclue à titre conservateur comme transition,
qu'elle soit capturée avant ou après T0. Ce choix ne dépend d'aucun résultat du 25.
Le registre original et le registre A01_EXPLORATORY restent intégralement conservés.
Aucun registre au nouveau préfixe n'existait lors du contrôle.

## Démarrage contrôlé

Après T0 : premier run PROSPECTIVE avec initialize=true ; ensuite initialize=false.
Un registre sans capture admissible est un état normal « activé, en attente de
première observation », pas une validation prédictive. Vérifier la sauvegarde et
la restauration même de ce registre initial vide d'observations.
Puis vérifier le premier import réel, ses horaires, identités, rangs et absence de
doublons, et un événement automatique distinct d'un test manuel.

Si la configuration ou la fusion prend du retard au-delà du T0 prévu : ne pas
lancer l'étude avec cet enregistrement ; conserver l'historique du report et
préinscrire une nouvelle échéance future avant toute observation admissible.
Aucune modification ultérieure de start.json ne doit antidater la préinscription.

Les règles A01 approuvées restent inchangées : deux seuils 98 %, maturité final
ou début archivé +12h, fenêtre fixe 180 jours, minima statistiques, corrections
soumises à revue autorisée et aucune promotion automatique. Aucun fichier du
manifeste de version n'est modifié par cette préinscription.
