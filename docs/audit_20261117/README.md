# HENACHEL — audit du 17 novembre 2026

## État du livrable

Correctif préparé ; suite complète : 482 tests réussis avec LOKY_MAX_CPU_COUNT=1, plus 2 tests de métriques ajoutés ensuite et réussis ; aucune nouvelle probabilité brute ni aucun modèle activé. **La remise en production n'est pas certifiée : aucune nouvelle exécution distante du correctif n'a été vérifiée.** Les métriques ci-dessous concernent les sélections canoniques récupérables, pas tout l'univers prédit. Aucun Excel publié.

## Incident et causes vérifiées

- Run [44104076189](https://github.com/mazalazop/nhl-value-betting-model/actions/runs/44104076189), 17 novembre : contexte exécuté, POINT ignoré, deux artefacts contexte/historique, aucun artefact modèle. La condition POINT exclut explicitement le cron `17 * * * *`. Le nom global et le statut success ne distinguaient pas cette actualisation d'une production. Ce run ne prouve pas un échec du modèle.
- Run [43858686077](https://github.com/mazalazop/nhl-value-betting-model/actions/runs/43858686077), 16 novembre 11:32 UTC : même situation. La proximité de 12h30 Paris n'en fait pas un run de production.
- Run [43957424774](https://github.com/mazalazop/nhl-value-betting-model/actions/runs/43957424774), déclenché par le dispatcher le 16 novembre à 21:10 UTC : POINT exécuté, sauvegarde exécutée. Les logs indiquent 96 POINT, 96 GOAL et 89 observations bookmaker, sur deux matchs. Artefacts modèle 14278390302 et historique 14276548221. Le contenu binaire du ZIP n'a pas été contrôlé localement : ces logs et métadonnées ne sont pas une certification complète.
- Deux horaires quotidiens coexistaient : 11:30 UTC dans le workflow global (13:30 Paris en été), et 12:30 Europe/Paris dans le dispatcher. Les deux chemins pouvaient produire deux calculs. Les modifications des fichiers workflows sur main déclenchaient également des calculs.
- GOAL et sa capture prospective toléraient les erreurs. Un success global ne garantissait ni leur présence ni leur fraîcheur. Le backup planifié utilisait une variable différente de celle qui active effectivement GOAL.
- Les workflows BUT shadow sont aussi déclenchés par les runs de contexte. Le source job du run 43969583323 cite explicitement l'entrée 43967717269 et `eligible=false` : absence d'artefact modèle attendue, pas panne du modèle. A01 peut ignorer l'évaluation lorsque son stage ne trouve aucune nouvelle entrée. Ces études ne doivent pas être forcées à recalculer une date déjà capturée.

Le retard observé du dispatcher ne permet pas d'attribuer une cause interne précise à GitHub. La documentation [schedule](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule) autorise retards et suppression de tâches ; aucune garantie de démarrage à la minute ne peut être donnée avec ce seul service.

## Correctif proposé

1. Un seul cron de production : 12:30, timezone IANA Europe/Paris. Le cron de contexte reste identifiable, avec un titre distinct. Dispatcher manuel uniquement ; suppression des productions automatiques sur modification de YAML.
2. Reçu de production par date NHL et branche. Un reçu d'une exécution réussie vérifiant les marchés requis évite un second calcul. Une erreur API bloque la décision ; elle n'autorise pas un recalcul aveugle. La concurrence historique reste sérialisée.
3. Contrôle avant publication : fichiers créés pendant l'exécution, date, identités uniques, probabilités finies entre 0 et 1, matchs non commencés, même univers POINT/GOAL, compte POINT conforme au résumé, prix Unibet admissibles et pré-match. Manifeste SHA-256. Échec si GOAL/capture prospective requis ont échoué. GOAL conserve son statut expérimental.
4. Job final qui échoue si POINT ou la sauvegarde requise ne réussissent pas. Un reçu précédent valide est signalé explicitement comme réutilisé. Sauvegardes historiques conservées même après échec.
5. Plancher demandé de **1,35 inclus** dans la sélection, l'import longitudinal et ses métriques ; README et tests de frontière cohérents. Aucun changement du classement probabiliste ou des probabilités.

Limites opérationnelles : les jours sans matchs sont bloqués explicitement par le contrôle actuel, à traiter comme un statut « sans matchs » vérifié dans une évolution dédiée. Un reçu reste une preuve d'une livraison antérieure, pas une garantie de disponibilité actuelle de tous ses artefacts. La concurrence GitHub et les retards schedule nécessitent encore un contrôle réel ; une horloge externe indépendante serait nécessaire pour un engagement horaire strict. Le correctif ne réactive aucune publication GOAL.

## Bilan observé disponible

Source : 41 fichiers canoniques, du 3 octobre au 16 novembre, figés au commit `da437adf8c6a08e35c81492897b8405a84917d04`. [Provenance et preuves des runs](provenance.json), [toutes les métriques](observed.json). L'historique canonique lui-même et les archives prospectives complètes n'ont pas pu être téléchargés dans le processus local : le réseau du shell ne résout pas GitHub. Le connecteur donne accès aux fichiers texte et aux métadonnées ; les ZIP renvoyés comme liens ne sont pas des données analysées localement.

| Mesure POINT | Valeur |
|---|---:|
| Sélections proposées | 444 |
| Réglées / non réglées | 440 / 4 |
| Succès | 259 |
| Réussite | 58,86 % |
| Probabilité annoncée moyenne, réglées | 60,63 % |
| Écart annoncé − observé | +1,76 point |
| Brier / log-loss | 0,237549 / 0,668129 |
| Profit simulé, 1 unité par sélection réglée | −44,05 unités |
| ROI simulé | −10,01 % |
| IC descriptif ROI, blocs de 7 dates | [−17,32 % ; −2,52 %] |
| EV moyenne annoncée | −6,76 % |

Résultats NHL observés ; mises et exécution des paris inconnues. ROI **simulé**, pas bénéfice réel du compte. Les intervalles exploratoires utilisent 10 000 tirages, graine fixe, blocs circulaires de dates ; aucune correction de multiplicité et aucune validation confirmatoire de stratégie. Segments hebdomadaires, probabilités, cotes et edge dans le JSON. Les groupes vides ne sont pas des résultats à zéro.

| Classement par probabilité dans les picks disponibles | Réglés | Réussite | ROI simulé |
|---|---:|---:|---:|
| Top 3 | 121 | 65,29 % | −3,93 % |
| Top 5 | 203 | 63,55 % | −5,71 % |
| Top 10 | 406 | 60,10 % | −8,29 % |

Ces classements sont des simulations **à l'intérieur des picks déjà proposés**. Ils ne reconstituent pas le Top K du modèle sur tous les joueurs. Les non-réglés ne sont pas remplacés pour améliorer les résultats. Les tris edge et EV, publiés dans le JSON, ne montrent pas de rentabilité positive sur cette population.

Seuils : 1,35 et 1,40 donnent ici les mêmes 440 réglés et −10,01 %. À 1,50 : 224 réglés, −12,00 % ; à 1,60 : 117 réglés, −16,79 %. **L'ancien import excluait les prix sous 1,40 : cette égalité ne démontre rien sur le rendement de nouvelles sélections entre 1,35 et 1,40.** Aucun prix n'a été reconstruit.

GOAL : aucune ligne dans ce sous-ensemble canonique de paris. Cela ne signifie pas zéro prédiction GOAL. Les données des registres expérimentaux doivent être récupérées séparément pour actualiser les métriques ; les chiffres de l'étude précédente ne sont pas ceux de toute la saison au 17 novembre.

## Expériences existantes et recommandations

L'étude [V3 existante](https://github.com/mazalazop/nhl-value-betting-model/blob/433afd28dfbc1c0b1063e7ac6c4f5ee2e62f84be/research/v3/STATISTICAL_RESULTS.md) évite de refaire les mêmes expériences : 182 445 participations appariées par marché sur sept fenêtres chronologiques ; expérience tirs sur 58 169 participations. Les résultats sont des replays, distincts des 440 sélections observées ci-dessus.

- **Démontré dans les expériences antérieures** : retirer la calibration existante dégrade les pertes. Les familles historique offensif, utilisation et volume contribuent au candidat logistique. Cela ne démontre pas la supériorité de ce candidat face à la production.
- **Prometteur, non validé** : tentatives/proxys géométriques pour GOAL. Gain log-loss face au logit simple −0,002334, IC ajusté [−0,003346 ; −0,001286] ; face à GOAL V1, +0,000139 [−0,002427 ; +0,003361], donc aucune promotion.
- **Non concluant** : recalibration supplémentaire POINT, changements de rôle approchés par TOI/PP, contexte simplifié, séries chaudes et déficit de conversion. Pas de bonus automatique de disette. La contribution des disettes au candidat GOAL ne prouve pas un gain face à V1.
- **Non disponible / incomplet** : véritables xG point-in-time, historique PP1 confirmé et partenaires de ligne, absences et gardiens annoncés. Fenêtre 20 matchs et tableaux complets de disettes 1 à 5 non nouvellement évalués dans cet audit. Les proxys géométriques ne sont pas des xG.
- **Priorité pratique** : conserver les modèles ; archiver tous les candidats et cotes horodatées avant de tester une politique value indépendante. La forte probabilité de réussite ne garantit pas l'EV positive. Les formules utilisées sont `1 / cote`, `edge = p - 1/cote`, `EV = p*cote - 1`.

## Reproduction et travaux restants

Télécharger les fichiers listés dans provenance.json au commit indiqué dans un dossier isolé, puis :

```sh
.venv/bin/python scripts/audit_observed_season.py --inputs outputs/audit_20261117/inputs --output docs/audit_20261117/observed.json
.venv/bin/python -m pytest tests/test_production_delivery.py tests/test_production_receipt.py tests/test_strict_min_odds.py tests/test_runtime.py tests/test_postgame_quality.py -q -W error
```

Les CSV d'entrée sont dans le dossier local isolé `outputs/audit_20261117/inputs`. Les hashes git d'origine et SHA-256 des entrées analysées sont conservés séparément. Aucun fichier canonique distant n'est modifié par l'analyse.

Restent requis avant de déclarer la mission terminée : exécution distante du correctif ; téléchargement puis contrôle du ZIP POINT/GOAL et des sauvegardes ; mise à jour exhaustive des registres GOAL et de l'univers POINT ; validation des expériences manquantes ; vérification du modèle Excel exact. Le classeur `NHL_Henachel_2026-10-20_REMISE_EN_FORME.xlsx` n'a pas été trouvé dans le dépôt, les dossiers locaux examinés ou la recherche Drive (nom exact puis Henachel). Aucun remplacement approximatif ni classeur périmé n'a été produit. Ses quatre onglets et sa mise en forme doivent être vérifiés sur l'original avant tout livrable.
