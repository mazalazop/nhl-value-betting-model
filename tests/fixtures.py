import numpy as np
import pandas as pd


def games_fixture():
    rows = []
    for year in (2023, 2024):
        for n in range(25):
            dt = pd.Timestamp(year=year, month=10, day=1) + pd.Timedelta(days=n * 2 + (12 if n >= 12 else 0))
            away = 'MTL' if year == 2023 else 'NYR'  # player 2 transferred
            gid = year * 10000 + n
            for pid, team, opp, home in [(1, 'TOR', away, 1), (2, away, 'TOR', 0)]:
                goals, assists = int(n % 3 == 0), int(n % 2 == 0)
                rows.append(dict(id_match=gid, id_joueur=pid, date_match=dt,
                    saison=int(f'{year}{year+1}'), season_source=f'{year}{year+1}',
                    team_player_match=team, adversaire_match=opp, is_home_player=home,
                    id_equipe_domicile='TOR', id_equipe_exterieur=away,
                    buts_domicile=3, buts_exterieur=2, status='OFF',
                    buts=goals, passes=assists, points=goals+assists,
                    tirs=np.nan if n % 7 == 0 else float(n % 5),
                    temps_de_glace=16.0 + n % 4,
                    temps_pp=np.nan if n % 4 == 0 else 2.0,
                    plus_moins=0, penalty_minutes=0,
                    match_trouve=1, check_team_ok=1, check_opp_ok=1,
                    nom=f'Player {pid}', position='C'))
    return pd.DataFrame(rows)


def standings_fixture(games):
    rows = []
    for dt, season in games[['date_match', 'season_source']].drop_duplicates().itertuples(index=False):
        for team in ['TOR', 'MTL', 'NYR']:
            rows.append(dict(date_snapshot=dt-pd.Timedelta(days=1), api_date=dt-pd.Timedelta(days=1),
                standings_lookup_date=dt-pd.Timedelta(days=1), season_id=int(season),
                team_abbrev=team, conference_abbrev='E', division_abbrev='A',
                games_played=5, games_remaining=77, points=6,
                conference_sequence=8, division_sequence=4, wildcard_sequence=2,
                conference_cutoff_points=6, wildcard_distance=0, point_pctg=.6,
                goal_differential=1, l10_points=6))
    return pd.DataFrame(rows)
