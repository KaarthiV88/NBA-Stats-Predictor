import nba_api
import numpy
import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog as pgl

def get_player_id(playerName):
    all_players = players.get_players()
    
    for p in all_players:
        if(p['full_name'].lower() == playerName.lower()):
            return p['id']
        
    return ValueError(playerName, "not found!")

def get_head_to_head_stats(player_id, opponent_abbreviation, seasons=['2023-2024', '2024-2025'], season_type='Regular Season'):
    all_games = []
        
    for season in seasons:
        gamelog = pgl.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star=season_type
        )
        df = gamelog.get_data_frames()[0]


        head_to_head = df[df['MATCHUP'].str.contains(opponent_abbreviation, case=False)]
        head_to_head['SEASON'] = season
        all_games.append(head_to_head)


    if all_games:
        combined = pd.concat(all_games).reset_index(drop=True)
        return combined
    else:
        return pd.DataFrame() 