# Protocole prospectif BUT V2 — validé avant activation

Version `BUT_V2_PROSPECTIVE_1`, configuration `configs/but_v2_prospective.json`.
T0 sera la première journée NHL capturée après autorisation; aucune date passée
ne peut être ajoutée. Les résultats de développement ne sont pas prospectifs.

## Population et gel

Une paire complète par date NHL : première capture réussie de tous les joueurs
V1 du run global validé. Identités, équipes et coups d'envoi doivent coïncider.
V1 doit avoir été émis depuis moins de deux heures, avant chaque match; V2 doit
finir avant le premier match. Pas de sélection des seuls matchs encore futurs
après un retard, ni remplacement d'un absent dans le Top10. Une journée manquée
reste manquée. Les doublons de déclenchement conservent la première paire.
Les deux émissions ne sont pas simultanées : mesurer leur décalage, plafond deux
heures au début du fit; aucune nouvelle information de contexte ne modifie V2.

V1 = GOAL17_OPERATIONAL_REFIT existant; V2 = logistique définie avant le replay.
Aucun ajustement des variables, hyperparamètres ou règles pendant ce cycle.
Les coefficients sont réentraînés chaque jour sur le passé final disponible,
comme prévu par les algorithmes gelés. Au moins 160 dates historiques, dernière
observation globale à sept jours au maximum; ancienneté conservée par capture.
Les hashes algorithmes/configuration/dépendances sont verrouillés à T0; une
dérive stoppe la capture. Un changement nécessite un nouveau protocole/version,
pas une concaténation silencieuse des résultats.

## Calendrier de décision

- J30 (jours calendaires à partir de T0) : premier bilan descriptif obligatoire.
  Vérifier couverture, données manquantes, erreurs d'identité, retards, gardes
  temporels et calibration. Aucune promotion, aucun tuning sur ces résultats.
- Une seule évaluation confirmatoire à J180 sur les dates T0 à T0+179 incluses.
  Les observations suivantes restent descriptives et ne déplacent pas cette
  fenêtre. Si l'effectif ou la qualité sont insuffisants : INCONCLUANT, maintien
  en expérimentation. Toute nouvelle étude est enregistrée séparément.
- Les rapports quotidiens servent à surveiller les opérations; leurs fluctuations
  ne déclenchent ni arrêt pour succès ni promotion anticipée.

## Conditions nécessaires, jamais suffisantes à elles seules

Au moins 90 dates NHL distinctes, 20000 participations appariées **soldées**,
2000 résultats BUT positifs, 98% de settlement observé dans l'univers capturé.
Au moins 95% des journées V1 éligibles capturées. Réconcilier également le
calendrier et les runs V1 avec les journaux : l'absence d'un déclenchement shadow
ne doit pas disparaître du dénominateur. Le compteur fourni par l'outil porte
sur les jours rencontrés; il ne remplace pas cet audit externe.

Ces nombres sont des planchers opérationnels, pas une preuve de puissance.
Le replay n'a pas démontré de gain global et ses intervalles restent larges.
Ne pas promettre une promotion à J180. Le 22 octobre 2026, l’utilisateur a validé le gain
minimal utile (.001 log-loss et .0003 Brier, fixés ici) et la fenêtre de 180 jours;
si une analyse de puissance indépendante impose davantage d'observations,
modifier le protocole **avant T0**, jamais en fonction des gains prospectifs.

Gates statistiques à J180 : deltas V2−V1 ≤ −.001 en log-loss et ≤ −.0003 en
Brier; bornes supérieures des IC simultanés 97,5% de chaque delta <0 (Bonferroni
pour les deux critères), bootstrap apparié de blocs de sept dates NHL, 10000
réplications et seed 20261020. Les rapports exploratoires actuels utilisent
2000 réplications/IC95%; ils ne satisfont pas à eux seuls ce gate final.
Comparer saison régulière/playoffs et domicile/extérieur, drift par mois,
calibration par tranches de probabilités fixes et couverture des nouveaux joueurs.
Une dégradation robuste d'un segment suffisamment représenté bloque la promotion;
segment peu représenté = limite explicite, pas preuve d'équivalence.

Calibration : ECE sur déciles ≤ .03 et absence de dégradation > .005 contre V1,
avec effectifs de chaque intervalle; ne pas interpréter une tranche minuscule.
Top10 : taux parmi les sélectionnés soldés, nombre inconnu/absent séparé;
classement figé avant match. Top10 et ROI sont secondaires, ne peuvent sauver
un échec sur Brier/log-loss. Pas de choix rétrospectif du meilleur modèle.
Toute promotion exige en plus une décision humaine explicite.

## Observations, prix et calculs

Un résultat exige boxscore NHL final, ID match/joueur/équipe concordants et
participation observée (TOI >0). Absent, TOI nul/inconnu ou match non final :
UNRESOLVED; jamais LOSS. VOID n'est pas déduit sans règle bookmaker et preuve.
Recollecte des résultats sur sept jours, versions brutes conservées; après,
résultat final mis en cache. Corrections officielles tardives nécessitent une
réconciliation versionnée supplémentaire avant décision.

Cotes : seulement GOAL 1+, ID et horaire concordants, source et capture antérieures
à émission V2/match. Même prix archivé pour comparer les sélections. ROI à mise
unitaire du Top10 de chaque modèle, avec effectifs cotés et non cotés, purement
simulé; pas de prix reconstruit après match, pas de filtre value optimisé après coup.
Pas d'IC journalier artificiellement précis : pas d'intervalle de blocs avant
14 dates. Le rapport cumulatif fournit Brier, log-loss, calibration, classement,
coverage et différences appariées; les décisions restent manuelles.

## Enregistrement avant T0 — 22 octobre 2026

Autorisation utilisateur : intégration et activation shadow seulement. Le candidat
reste la logistique complète d'origine, sans sélection fondée sur le replay.
Les variables brutes temps PP historiques font déjà partie du candidat; aucun
nouveau temps PP projeté, gardien, PP1 ni xG n'est ajouté. La calibration est
sélectionnée chronologiquement à chaque refit selon l'algorithme gelé.
Les fichiers et variables sont inscrits dans `PRE_T0_REGISTRATION.json`.

Top10 quotidien pour chaque modèle : probabilité décroissante, puis identifiant
match croissant et identifiant joueur croissant pour départager les égalités.
La référence V1 est la probabilité GOAL17 du même artefact, jamais recalculée.
Le décalage entre les émissions est conservé; l'âge V1 est vérifié au début du fit
et l'ensemble des matchs doit encore être futur à la fin du fit V2.
Ni exclusion d'un joueur peu probable ni remplacement après observation du résultat.
La calibration utilise les dix intervalles fixes [0,.1), …, [.9,1].
La mise simulée est une unité pour chaque membre coté du Top10 figé, sans
filtre de value; les non-cotés et non-résolus restent comptabilisés séparément.
