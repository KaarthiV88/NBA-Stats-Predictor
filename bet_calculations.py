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
    # Get opponent stats (what the team allows to others)
    opp_stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star=season_type,
        measure_type_detailed_defense='Opponent'
    ).get_data_frames()[0]

    # Get advanced defensive rating
    adv_stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star=season_type,
        measure_type_detailed_defense='Advanced'
    ).get_data_frames()[0]

    # Filter for team
    opp_row = opp_stats[opp_stats['TEAM_ID'] == team_id]
    adv_row = adv_stats[adv_stats['TEAM_ID'] == team_id]

    if opp_row.empty or adv_row.empty:
        return None

    # Build defense DataFrame
    defense_df = pd.DataFrame({
        'TEAM_ID': [team_id],
        'GP': opp_row.iloc[0]['GP'],
        'Opp_PTS': opp_row.iloc[0]['PTS'],
        'Opp_REB': opp_row.iloc[0]['REB'],
        'Opp_AST': opp_row.iloc[0]['AST'],
        'Opp_FG_PCT': opp_row.iloc[0]['FG_PCT'],
        'DEF_RATING': adv_row.iloc[0]['DEF_RATING']
    })

    return defense_df

def get_offensive_team_stats(team_id, season="2024-25", season_type="Regular Season"):
    # Get base stats
    base_stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star=season_type,
        measure_type_detailed_defense='Base'
    ).get_data_frames()[0]

    # Get advanced stats
    adv_stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star=season_type,
        measure_type_detailed_defense='Advanced'
    ).get_data_frames()[0]

    # Filter for team
    base_row = base_stats[base_stats['TEAM_ID'] == team_id]
    adv_row = adv_stats[adv_stats['TEAM_ID'] == team_id]

    if base_row.empty or adv_row.empty:
        return None

    # Build offense DataFrame
    offense_df = pd.DataFrame({
        'TEAM_ID': [team_id],
        'GP': base_row.iloc[0]['GP'],
        'PTS': base_row.iloc[0]['PTS'],
        'REB': base_row.iloc[0]['REB'],
        'AST': base_row.iloc[0]['AST'],
        'FG_PCT': base_row.iloc[0]['FG_PCT'],
        'OFF_RATING': adv_row.iloc[0]['OFF_RATING']
    })

    return offense_df
