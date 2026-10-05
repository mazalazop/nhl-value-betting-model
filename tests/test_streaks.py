import numpy as np
import pandas as pd
from henachel.features import compute_streak_window_features


def reference_streaks(df):
    rows=[];current_hit=current_miss=0;previous=None
    years=df.season_start_year.tolist();hits=df.a_marque_un_point.tolist()
    for i,(year,hit) in enumerate(zip(years,hits)):
        if year!=previous:current_hit=current_miss=0
        history=[h for h,y in zip(hits[:i],years[:i]) if y>=year-1]
        run_hit=run_miss=max_hit=max_miss=count=0
        for value in history:
            run_hit=run_hit+1 if value else 0;run_miss=run_miss+1 if not value else 0
            max_hit=max(max_hit,run_hit);max_miss=max(max_miss,run_miss);count+=int(run_hit==5)
        rows.append([current_hit,current_miss,max_hit,max_miss,count,len(history)])
        current_hit=current_hit+1 if hit else 0;current_miss=current_miss+1 if not hit else 0;previous=year
    return np.array(rows)


def test_streaks_match_bruteforce_across_seasons_and_gaps():
    rng=np.random.default_rng(3)
    for hits in [rng.integers(0,2,400),np.ones(400,dtype=int),np.zeros(400,dtype=int)]:
        frame=pd.DataFrame({'date_match':pd.date_range('2020-01-01',periods=400),'id_match':range(400),'a_marque_un_point':hits,'season_start_year':np.repeat([2020,2021,2023,2024],100)})
        np.testing.assert_array_equal(compute_streak_window_features(frame).to_numpy(),reference_streaks(frame))
