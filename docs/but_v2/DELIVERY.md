# Livraison ASTRA BUT V2 — 20 octobre 2026

**Recommandation : conserver BUT V1; maintenir BUT V2 en expérimentation.**
Aucun remplacement ni fusion dans main effectué.

Quatre PR en brouillon, avec dépendances explicites :

1. [#9 — qualité des données](https://github.com/mazalazop/nhl-value-betting-model/pull/9), base main.
2. [#10 — modèle indépendant](https://github.com/mazalazop/nhl-value-betting-model/pull/10), base PR9.
3. [#11 — validation scientifique](https://github.com/mazalazop/nhl-value-betting-model/pull/11), base PR10.
4. [#12 — shadow et reporting](https://github.com/mazalazop/nhl-value-betting-model/pull/12), base PR11.

Elles doivent être revues dans cet ordre; retargeter les suivantes après
intégration de leur base. Elles ne sont pas quatre modifications concurrentes
de main. Chaque description détaille tests, limites et retour arrière.

## Ce qui est livré

- [Audit et qualité](AUDIT.md) : architecture réellement trouvée, données,
  workflows, synchronisation, limites et priorités. Corrections ID NHL,
  inconnus, doublons et preuves pré-match. Joueur avec TOI officiel nul :
  UNRESOLVED, jamais LOSS. Stützle et les archives existantes préservés.
- [Modèle](MODEL.md) : logistique, Poisson et HGB indépendants, configuration
  figée, calibration temporelle et tests anti-fuite.
- [Protocole](VALIDATION.md) et [résultats](RESULTS.md) : replay effectivement
  exécuté, 61950 lignes test sur 290 dates et 1721 matchs, cinq fenêtres,
  ablations et intervalles appariés. Aucun gain global crédible établi.
- [Shadow](SHADOW.md) : captures pré-match protégées, traçabilité, modèle figé,
  rapprochement et reporting comparatif, cotes optionnelles vérifiées et ROI
  exclusivement simulé. Workflow séparé manuel/réutilisable, non activé.

## Ce qui demeure limité

L'historique utilisé est rétrospectif et déjà exploré. Les dates exactes de
publication des corrections NHL ne sont pas connues. Il ne constitue pas une
validation prospective, même si le calcul des variables est chronologique.
Pas de nouvel historique fiable pour xG individuel, gardien ou composition
PP1/PP2. Pas de ROI historique calculable. Les proxys géométriques ne sont pas
des xG. Les résultats officiels existants n'ont pas été recollectés pendant
cette étude; leur provenance est celle du snapshot audité.

Les deux horaires postgame programmés ne garantissent pas deux exécutions
ponctuelles; les logs Actions montrent des retards. Le collecteur peut encore
recevoir une réponse NHL partielle ou une correction tardive. Ces risques
nécessitent surveillance et conservation des sources brutes.

Le shadow fonctionne sur fixtures techniques. Aucun nouveau run shadow réel,
pari, publication Sheets ou activation quotidienne n'est revendiqué. Son
artefact d'entrée doit être alimenté et sauvegardé durablement avant suivi
réel. Le workflow réutilisable permet ensuite un raccordement quotidien,
avec une défaillance indépendante de POINT/V1.

## Contrôles d'intégrité

État initial Git propre, branche de travail dédiée; branche utilisateur
préservée. Diff nul sur modèles POINT/BUT V1, features partagées, workflow
global, archives postgame et revue Stützle. Les hashes des fichiers de recherche
exécutés correspondent encore au protocole du replay. Pas de retuning après
lecture des résultats. Les sources originales compressées sont conservées
localement avec vérification du SHA256 après décompression.

La suite stricte a d'abord révélé un descripteur de fichier oublié dans un
nouveau test. Le défaut a été corrigé, sans masquer les warnings. Les CI des
PR sont la référence actualisée pour la vérification distante.

## Décisions humaines

Revoir les quatre PR et choisir leur intégration, sans promotion de modèle.
Autoriser séparément le raccordement shadow et son contrat de données.
Fixer ensuite une période prospective indépendante, la taille d'échantillon
et les critères de promotion avant d'observer ses résultats. Aucune de ces
décisions n'est prise automatiquement par le code livré.
