# Protocole structurel figé — POINT et BUT

Le nom du fichier est conservé pour les liens ; **ce protocole est désormais préenregistré**, avant tout fit structurel. `protocol.json` est la spécification des listes exactes, inputs hashés, paramètres, folds et critères. Aucun changement de production.

## Population et budget

2022-23 + 2023-24, saison régulière, toutes les participations sans filtrage selon disponibilité. 2022-23 fournit le premier historique ; N−1 manque sur cette saison d'entraînement, reste NaN. N−2 est indisponible, donc exclu. Quatre fenêtres de développement 2023-24, définies dans le JSON. Les anciens holdouts déjà consommés sont explicitement du développement, **jamais une confirmation indépendante**. Pas de saison supplémentaire ni de tuning.

52 fits HGB depth4 au maximum : V0–V5 × 4 folds × POINT/BUT, plus C19 × 4 folds. Pas d'arrêt anticipé ni de combinaison a posteriori. Baselines sauvegardées une seule fois. Modèle/calibration entraînés seulement avant le fold ; 85 % des dates passées pour fit, 15 % pour calibration temporelle commune. Features constantes sélectionnées sur fit seulement ; NaN laissés au HGB.

## Définitions avant résultats

H : pour points, buts, tirs et indicateurs 1+, moyennes N−1 complet et saison courante strictement antérieure. Exposer les deux mélanges `(somme_courante + k*moyenne_N−1)/(n_courant+k)`, k=10 et k=30, sans sélectionner k après résultats. Ce sont deux hypothèses de pseudo-exposition, pas des poids estimés ni une vérité hockey. Si N−1 inconnu, mélange inconnu ; la moyenne courante reste séparément disponible. Compteurs de matchs, domicile, production/60 = 60 × somme production / somme minutes, uniquement sur observations appariées connues. Aucune imputation zéro.

R : moyennes TOI et PP N−1/courantes, usage/tirs récents 5/10, écart 5–10 TOI/PP et écart-type PP10 (min5). Part PP = somme minutes PP joueur / somme minutes PP équipe sur participations appariées connues ; temps équipe officiel en secondes converti en minutes. Ce ratio n'est PAS PP1. Repos, back-to-back, calendrier/transfert observé identiques aux fonctions existantes. Présence future d'un joueur en équipe exige une preuve roster, jamais un roster rétrospectif présenté comme pré-match.

F : résultats 5/10 et streaks/drought existants ; plus `(5*moyenne5+10*moyenne_N−1)/15` et `moyenne5−moyenne_N−1` pour points/buts. La pseudo-exposition 5 doit être remplacée par le nombre réel de résultats connus parmi les cinq derniers matchs lorsqu'il est inférieur à cinq. Les poids sont fixés avant test, inconnus préservés. Le contraste V3−V1 et V4−V2 mesure conjointement la forme 5/10 ; il ne permet PAS de conclure séparément sur chaque fenêtre ni chaque feature.

S : moyennes cumulées saisonnières antérieures équipe PP opportunities/penalties drawn et adversaire penalties taken/times shorthanded ; efficacité PP = buts PP/opportunités, PK = 1−buts PP concédés/désavantages. Dénominateur nul = inconnu. Expected PP = moyenne arithmétique des taux PP équipe et désavantages adverses, deux taux obligatoires, minimum cinq matchs connus chacun. C'est une estimation non apprise, pas un événement futur réel. Une interaction : part PP courante joueur × expected PP. Aucun faux PP1, aucune assimilation pénalités/opportunités.

Toutes les statistiques sont strictement antérieures au jour du match ; aucune autre rencontre du même jour n'est utilisable. Les consommateurs futurs appellent les mêmes fonctions avec résultats masqués. Les données manquantes sont conservées ; nombre et taux de couverture sont rapportés.

## Comparaisons et garde-fous

V0 = POINT81 production ou GOAL17. V1 H, V2 H+R, V3 H+F, V4 H+R+F, V5 H+R+F+S. Les contrastes évaluent des blocs, pas une causalité ni une importance individuelle. V5 ne sépare pas le gain des pénalités de celui des autres special teams.

**Condition bloquante POINT** : obtenir les snapshots standings officiels J−1 via la fonction de normalisation production, vérifier date API/saison/identités. Ne pas tester une baseline privée silencieusement de ses standings. Les dates historiquement sans snapshot restent inconnues selon le contrat production, avec couverture explicite ; des réponses réseau défaillantes ne sont pas des absences métier. Si la source est indisponible, suspendre POINT sans modifier le protocole.

LL/Brier/AUC/AP/ECE, reliability bins, top décile, effectifs/dates/joueurs et diagnostics F/D. IC apparié bootstrap blocs de sept dates, 10 000 réplications seed20261004 ; correction dix comparaisons principales (5 par cible). Gain LL >=0,0005, Brier amélioré, les quatre folds favorables en LL/Brier, borne supérieure IC corrigé <0 et ECE ne se dégrade pas de plus de0,005. BUT doit également satisfaire les comparaisons face à C19 et joueur lissé (même correction conservatrice). Benchmark joueur : `(hits_saison_pre+10*prévalence_train)/(matchs_saison_pre+10)`. Aucun ROI sans cotes fiables.

## Confirmation prospective et PP1

Aucun résultat historique de cette étude n'autorise une promotion. Candidat éventuellement retenu verrouillé avant 2026-11-01 UTC ; validation fixe du 2026-11-01 au 2027-04-19 inclus. Au moins90dates/20000participations ; sinon non concluant, sans allonger après consultation. Prédictions archivées avant début réel, modèle/calibrateur/features/code hashés et immuables. Une seule ouverture finale ; mêmes observations pour benchmarks. Pas de retuning ni lecture intermédiaire pour sélection. La préparation du mécanisme ne prouve pas sa collecte effective.

PP1 prospectif reste séparé : source identifiable, horodatage source et collecte UTC, match/date, équipe/joueur NHL, unité explicitement observée ; absence UNKNOWN jamais PP2. Faute de source vérifiée, seul le contrat est documenté. Aucun filtre de pari modifié, aucun modèle défenseur créé.
