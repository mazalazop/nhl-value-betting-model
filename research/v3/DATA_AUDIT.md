# Inventaire V3 — sources figées au 1er novembre 2026

Le périmètre analysé est celui des fichiers effectivement récupérés, pas une promesse de six saisons complètes. `results/inventory.json` donne les identités, dates et effectifs exacts ; `results/missingness.csv` expose les 149 variables pour chaque saison. La capture de l'audit est horodatée dans ces fichiers. Les données postgame publiées s'arrêtent au 31 octobre ; l'historique large au 30 octobre.

| Saison | Observations | Matchs | Joueurs | Période |
|---|---:|---:|---:|---|
| 2021–22 | 50 415 | 1 401 | 1 006 | 12/10/2021–26/06/2022 |
| 2022–23 | 50 381 | 1 400 | 953 | 07/10/2022–13/06/2023 |
| 2023–24 | 50 389 | 1 400 | 925 | 10/10/2023–24/06/2024 |
| 2024–25 | 50 321 | 1 398 | 924 | 04/10/2024–17/06/2025 |
| 2025–26 | 50 183 | 1 394 | 940 | 07/10/2025–14/06/2026 |
| 2026–27, partielle | 7 776 | 216 | 733 | 29/09/2026–30/10/2026 |

Total : **259 465 observations, 7 209 matchs, 1 511 joueurs**, zéro doublon joueur-match. Ces nombres décrivent la source, avant les exclusions de recherche : matchs finaux de saison régulière/playoffs et participants avec TOI positif. Cinq saisons sont complètes ; la sixième ne l'est pas. Les données locales initiales (104 284 lignes) sous-estimaient la profondeur disponible dans la sauvegarde durable.

## Admissibilité et limites temporelles

| Source | Utilisation autorisée dans cette recherche | Limite |
|---|---|---|
| Historique large restauré et vérifié par SHA-256 | Entraînement et replay chronologique exploratoires | Statistiques finales reconstruites ; dates originales des corrections non connues. Pas une certification point-in-time historique. |
| Registre CANONICAL POINT | Scores/ROI descriptifs des sélections pré-match | Identités, probabilités, cotes et heures rapprochées de l'import original. Contrat du ledger accepté ; première publication indépendante non recertifiée. |
| Registres BUT original et A01 | Lecture des rapports prospectifs archivés, séparément | Chevauchement des joueurs/dates : interdiction de sommer les deux études. Aucune modification de leur règlement. |
| NHL play-by-play, 2 saisons | Géométrie/tentatives des matchs antérieurs, exploration | Récupéré en novembre 2026, pas à l'époque du match. Version des corrections indisponible historiquement. |
| CLASSEUR/EXTRA | Description distincte | Pas de preuve suffisante pour des scores probabilistes ou ROI comparables au CANONICAL. |

Chaque variable V3 est décalée d'un match au moins. La journée test n'entre ni dans l'entraînement ni dans la calibration. Les agrégats de saison sont calculés avec le passé seulement. L'imputation et la standardisation sont ajustées sur l'entraînement. Les anciens modèles sont appelés avec une date limite ; leur algorithme reste inchangé. Leurs probabilités recalculées sont des **références de replay**, pas des prédictions réellement émises à l'époque.

Diagnostic supplémentaire : les 244 309 dates de standings renseignées sont strictement antérieures à la date du match. Les fenêtres natives de tirs traversent les saisons, alors que les fenêtres du candidat V3 redémarrent à chaque saison. Leur comparaison par simple remise à zéro annuelle produit donc des différences attendues, pas une fuite. Les diagnostics de recalcul sont dans `native_temporal_diagnostics.json`. Cette asymétrie et les variables supplémentaires de la référence limitent l’interprétation causale de la comparaison d’architectures ; les ablations internes gardent la même convention.

Les périodes 2024–26 déjà étudiées restent exploratoires. Même une différence assortie d'un intervalle ajusté ne constitue pas une nouvelle validation indépendante. Les Top5/10 du replay sont calculés dans l'univers des participants connus a posteriori : ils ne certifient pas la couverture des absences d'une future liste pré-match.

## Variables réellement disponibles

Buts, passes, points, tirs cadrés, TOI, temps PP, identités stables, équipe/adversaire, domicile, dates, statut du match, production d'équipe et moyennes historiques sont disponibles dans la source large. Les manquants restent inconnus jusqu'à l'imputation d'entraînement ; un PP inconnu n'est pas un zéro de PP. Les premières apparitions de saison ont des historiques incomplets, visibles dans `main_feature_missingness.json`.

La collecte officielle couvre **2 792/2 792 matchs** des saisons 2024–25 et 2025–26. Les tirs cadrés correspondent à la source historique pour **100 449/100 504 lignes**. Les 55 autres lignes n'alimentent pas les proxys. Tous les comptes de buts de ces 100 504 lignes concordent avec le play-by-play récupéré. Les 22 échecs réseau/DNS initiaux ont été récupérés avec une reprise utilisant le cache ; leurs tentatives restent enregistrées localement.

La distance et l'angle concernent les tirs non contrés : la position d'un tir contré peut désigner l'endroit du contre. Un tir proche signifie ici distance ≤20 pieds, pas une occasion officiellement « haute dangerosité ». Le rebond est une approximation : tir suivant un arrêt de la même équipe dans les trois secondes. Au moins 90 % de coordonnées renseignées sont nécessaires au calcul géométrique par joueur-match. Les fenêtres précédentes demandent au moins trois observations connues sur dix ; leur couverture par fold figure dans `quality.json`. Un joueur sans tir n'a pas une distance moyenne nulle.

**Aucun xG individuel fiable avec version historique connue n'a été intégré.** Aucun statut PP1 confirmé, partenaire de ligne, retour de blessure confirmé, gardien pré-match, pénalité adverse ou déplacement documenté n'est utilisé comme s'il avait été connu. Les minutes PP et changements de minutes ne prouvent pas une composition PP1.

## Cotes, classeurs et sources externes

Les 174 cotes CANONICAL admissibles proviennent d’Unibet ; aucun comparateur FDJ admissible n’est présent dans cet échantillon. Les cotes du grand historique ne sont pas archivées avec une provenance pré-match : son ROI reste non calculable. Les politiques value ne sont testées que sur le sous-univers CANONICAL déjà sélectionné, avec cotes ≥1,40 et timestamps antérieurs au match. Les analyses par bookmaker, cote, edge, joueur, équipe et date sont fournies dans `canonical.json`. Les segments contextuels sans preuve pré-match ne sont pas reconstruits artificiellement.

Aucun classeur Excel quotidien exploitable n'a été trouvé dans le dépôt accessible. Les 28 sélections POINT et 28 BUT du groupe CLASSEUR existant sont des reconstructions distinctes. `snapshots.classeur()` fournit un normaliseur testé pour les quatre catégories demandées, avec déduplication d'un même pick et appartenance à plusieurs catégories. Ce prototype n'est pas une récupération de classeurs manquants.

MoneyPuck a été examiné, sans télécharger ni redistribuer ses données. Sa [page de données](https://moneypuck.com/data.htm) prévoit l'usage non commercial/journalistique avec attribution et une prise de contact pour les autres usages ; elle ne constitue pas une autorisation commerciale générale. Son [descriptif du modèle](https://moneypuck.com/about.htm) indique une refonte 2026–27 entraînée sur 2023–24 à 2025–26 : des xG recalculés avec cette version ne sont pas automatiquement admissibles pour prédire ces mêmes saisons. Les tirs contrés sont absents de son jeu de tirs annoncé. Une licence et une version historiquement admissible sont donc des prérequis distincts.

Les réponses NHL proviennent de `https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play`. Les enveloppes locales conservent URL, capture UTC et document. Les données brutes restent sous `outputs/`, hors Git ; seuls des agrégats et empreintes sont livrés. L'accès technique à une source ne vaut pas autorisation de redistribution commerciale.
