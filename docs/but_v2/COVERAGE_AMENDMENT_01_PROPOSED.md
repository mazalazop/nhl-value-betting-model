# Amendement de couverture A01 — historique de la proposition initiale

Version historique : A01-DRAFT-1, 23 octobre 2026. Le principe a depuis été approuvé.
La préparation concrète révisable est [A01_PROTOCOL_PREPARATION_1.md](A01_PROTOCOL_PREPARATION_1.md).
Son activation reste **EN ATTENTE DE VALIDATION HUMAINE** ; ce texte initial est conservé pour traçabilité.
Aucun fichier gelé, seuil, dénominateur officiel, résultat ni sélection ne change.
Ce texte n'est ni une nouvelle préinscription effective ni une autorisation de promotion.

## Problème constaté après T0

PROSPECTIVE_PROTOCOL.md exige 98 % de settlement dans l'univers capturé.
`but_v2_prospective.summarize` calcule WIN/LOSS divisés par toutes les lignes ;
`final_assessment` applique ce résultat au seuil 0.98, dans la fenêtre de 180 jours.
`but_v2_shadow.report` exige résultat final, buts connus et TOI > 0 ; absents,
TOI manquants/nuls et matchs non finaux restent UNRESOLVED. Le protocole protège
correctement contre les fausses défaites, mais la métrique agrège absence connue
et manque d'information. Il n'existe pas de statut statistique séparé de
non-participation dans cette métrique. VOID n'est pas inféré.

Le 22 octobre : N=539, S=425 participants réglés, A=114 non-participations
corroborées (absents du boxscore final ET des rosterSpots NHL), M=0 cas restant
inexpliqué dans cet audit. Causes individuelles des absences non établies.

| Diagnostic, sans remplacer la métrique officielle | Valeur |
|---|---:|
| Couverture officielle S/N | 425/539 = 78,8497 % |
| Part des non-participations corroborées A/N | 114/539 = 21,1503 % |
| Part réellement inconnue M/N, après fins officielles | 0/539 = 0 % |
| Participants identifiés dont résultat réglé S/P | 425/425 = 100 % |
| Information de participation/résultat documentée (S+A)/N | 539/539 = 100 % |

Les 114 restent UNRESOLVED dans le registre officiel : ce diagnostic ne les
convertit ni en LOSS ni en VOID. Les 86 paires pré-match du 23 octobre sont en
attente normale : ne pas les qualifier d'information manquante définitive.
À cet instant, le rapport cumulatif officiel reste 425/625 = 68 %.

## Atteignabilité

Pour cette cohorte fixe, la couverture officielle plafonne à 78,8497 % si les
114 non-participations sont définitives. Le seuil 98 % y est inatteignable.
Sur l'étude cumulée, il n'est pas mathématiquement impossible : avec seulement
ces 114 absences et aucun autre non-règlement, il faudrait N >= 5700, soit
5161 nouvelles paires toutes réglées après les 539 initiales. Si la proportion
d'absents reste proche de 21,15 %, même un collecteur parfait plafonnera près
de 78,85 %. Une journée ne démontre pas la proportion future. À univers inchangé,
98 % est donc structurellement incompatible avec un taux durable d'absents >2 %.
Ne pas masquer ce problème en sélectionnant après match les joueurs du dénominateur.

## Proposition pour une étude distincte, si approuvée

1. Conserver BUT_V2_PROSPECTIVE_1, son dénominateur, son T0 et tous ses rapports.
   L'évaluation originale reste soumise au gate original, même s'il échoue.
2. Ajouter une couche d'audit de participation, sans modifier les prédictions :
   PARTICIPANT_SETTLED, NONPARTICIPATION_CORROBORATED, PARTICIPANT_RESULT_MISSING,
   PARTICIPATION_UNKNOWN, GAME_PENDING. Définir à l'avance les preuves minimales,
   contrôles d'identité/équipe, fraîcheur, complétude des effectifs et gestion des
   reports/corrections. Absence d'une seule réponse ne suffit jamais à classer A.
   Conserver les sources horodatées et leurs révisions. Désaccord => inconnu.
3. Afficher systématiquement S/N, A/N, M/N et P/N. Sur l'univers arrivé à maturité,
   proposer un gate d'information complète (S+A)/N >= 0.98 ET un gate de résultats
   parmi les participants identifiés S/P >= 0.98. Afficher séparément les joueurs
   dont la participation est inconnue pour empêcher une amélioration artificielle
   de S/P. Aucun abandon silencieux de dates reportées à la clôture : inconnu.
   Ces nouveaux gates ne sont PAS des synonymes de l'ancien settlement à 98 %.
4. Garder le candidat et toutes les règles de sélection, les Top10 sans remplacement,
   les métriques sur les mêmes participants, les minima 90 dates/20000 résultats/
   2000 positifs et les exigences de gains/calibration/robustesse. Ne pas attribuer
   de rendement aux absents sans règles de règlement des cotes archivées.
5. Si ce changement sert à une décision confirmatoire, enregistrer avant application
   un nouveau protocole identifié, une nouvelle date T0' et une fenêtre de 180 jours.
   Les données déjà observées restent exploratoires pour cette nouvelle analyse ;
   ne pas présenter l'amendement post-T0 comme un critère préenregistré à T0.
   Maintenir séparément l'étude originale, sans réinitialiser ni écraser son registre.
6. Prévoir tests des cas absents, effectifs incomplets, TOI inconnu/nul, report,
   correction officielle, doubles identités, et conservation exacte des Top10.
   Versionner les nouveaux outils/configuration hors des fichiers gelés existants.

## Décision demandée

Valider ou refuser le principe d'une étude distincte avec ces définitions.
L'approbation de ce document seule ne déploie rien : l'implémentation, les tests,
le protocole complet et son T0' doivent être revus avant activation.
Sans approbation : poursuivre l'étude originale inchangée, en signalant le risque
que son gate de couverture rende la conclusion confirmatoire INCONCLUANTE.
