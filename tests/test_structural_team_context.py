import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from collect_structural_team_context import validate_records
GAMES={2022020001:dict(date_match='2022-10-07',season_source=20222023,id_equipe_domicile='NSH',id_equipe_exterieur='SJS')}
def rows():return [dict(gameId=2022020001,gameDate='2022-10-07',gamesPlayed=1,opponentTeamAbbrev=opp,teamId=n,ppOpportunities=0,powerPlayGoalsFor=0,ppTimeOnIcePerGame=0) for n,opp in enumerate(['NSH','SJS'],1)]
def test_complete_game_team_contract_and_true_zero():
    out=validate_records(rows(),'powerplay',GAMES)
    assert len(out)==2 and out.ppOpportunities.eq(0).all()
def test_missing_stays_unknown():
    r=rows();r[0]['ppOpportunities']=None
    assert validate_records(r,'powerplay',GAMES).ppOpportunities.isna().sum()==1
@pytest.mark.parametrize('column,value',[('gameId',2022020002),('gameDate','2022-10-08'),('gamesPlayed',82),('opponentTeamAbbrev','TOR'),('ppOpportunities',float('nan')),('ppOpportunities',-1)])
def test_incoherent_rows_refused(column,value):
    r=rows();r[0][column]=value
    with pytest.raises(ValueError):validate_records(r,'powerplay',GAMES)
def test_duplicates_and_incomplete_coverage_refused():
    with pytest.raises(ValueError):validate_records(rows()[:1],'powerplay',GAMES)
    with pytest.raises(ValueError):validate_records(rows()+rows()[:1],'powerplay',GAMES)
