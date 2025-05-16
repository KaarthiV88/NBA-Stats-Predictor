import nba_api
import numpy
import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog as pgl
from nba_api.stats.endpoints import teamdashboardbygeneralsplits
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.static import teams

def get_player_id(playerName):
    all_players = players.get_players()
    
    for p in all_players:
        if(p['full_name'].lower() == playerName.lower()):
            return p['id']
        
    return ValueError(playerName, "not found!")

def get_head_to_head_stats(player_id, opponent_abbreviation, seasons=['2023-2024', '2024-2025'], season_type="Regular Season"):
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

def get_team_id(teamAbbr):
    team = teams.find_team_by_abbreviation(teamAbbr)
    if(team):
        return team['id']
    else:
        return ValueError(teamAbbr, "team could not be found. Search up the official team abbreviation and try again!")


def get_defensive_team_stats(team_id, season = "2024-25", season_type="Regular Season"):
    dashboard = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
        team_id=team_id,
        season=season,
        season_type_all_star=season_type
    )

    overall_df = dashboard.get_data_frames()[0]  
    opp_df = dashboard.get_data_frames()[1]      

    def_rating = overall_df.iloc[0].get('DEF_RATING', None)

    defense_stats = opp_df[[
        'GP', 'W', 'L', 'W_PCT',
        'MIN', 'PTS', 'REB', 'AST', 'FG_PCT'
    ]].copy()

    defense_stats.rename(columns={
        'PTS': 'Opp_PTS',
        'REB': 'Opp_REB',
        'AST': 'Opp_AST',
        'FG_PCT': 'Opp_FG_PCT'
    }, inplace=True)

    defense_stats['DEF_RATING'] = def_rating

    return defense_stats.reset_index(drop=True)

def get_offensive_team_stats(team_id, season="2024-25", season_type="Regular Season"):
    dashboard = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
        team_id=team_id,
        season=season,
        season_type_all_star=season_type
    )

    team_stats = dashboard.get_data_frames()[0]  # Team’s own stats

    off_rating = team_stats.iloc[0].get('OFF_RATING', None)

    offense_stats = team_stats[[
        'GP', 'W', 'L', 'W_PCT',
        'MIN', 'PTS', 'REB', 'AST', 'FG_PCT'
    ]].copy()

    offense_stats.rename(columns={
        'PTS': 'Team_PTS',
        'REB': 'Team_REB',
        'AST': 'Team_AST',
        'FG_PCT': 'Team_FG_PCT'
    }, inplace=True)

    offense_stats['OFF_RATING'] = off_rating

    return offense_stats.reset_index(drop=True)
