import pandas as pd
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats
import time
import json

def get_player_id(player_name):
    """
    Fetches the player ID based on the player's full name.
    """
    try:
        player_dict = players.find_players_by_full_name(player_name)
        if not player_dict:
            raise ValueError(f"Player '{player_name}' not found.")
        return player_dict[0]['id']
    except Exception as e:
        raise ValueError(f"Error fetching player ID for '{player_name}': {e}")

def get_team_id(team_abbr):
    """
    Fetches the team ID based on the team's abbreviation.
    """
    try:
        team_dict = teams.find_team_by_abbreviation(team_abbr.upper())
        if not team_dict:
            raise ValueError(f"Team '{team_abbr}' not found.")
        return team_dict['id']
    except Exception as e:
        raise ValueError(f"Error fetching team ID for '{team_abbr}': {e}")

def get_head_to_head_stats(player_id, opponent_abbr, seasons=['2023-24', '2024-25'], season_type='Regular Season'):
    """
    Fetches the player's game logs against a specific opponent for the given seasons.
    """
    game_logs = []
    for season in seasons:
        try:
            gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season, season_type_all_star=season_type)
            df = gamelog.get_data_frames()[0]
            if df.empty:
                print(f"No games found for player ID {player_id} in season {season}, {season_type}")
                continue
            df['SEASON'] = season
            df['OPPONENT'] = df['MATCHUP'].apply(lambda x: x.split(' vs. ')[1] if ' vs. ' in x else x.split(' @ ')[1])
            head_to_head = df[df['OPPONENT'] == opponent_abbr]
            game_logs.append(head_to_head)
            time.sleep(1)  # Delay to avoid rate-limiting
        except json.decoder.JSONDecodeError as e:
            print(f"JSONDecodeError for season {season}: {e}. Skipping...")
            continue
        except Exception as e:
            print(f"Error fetching data for season {season}: {e}. Skipping...")
            continue
    if game_logs:
        return pd.concat(game_logs).reset_index(drop=True)
    else:
        print(f"No head-to-head games found for player ID {player_id} against {opponent_abbr}")
        return pd.DataFrame()

def get_defensive_team_stats(team_id, season='2023-24', season_type='Regular Season'):
    """
    Fetches the defensive stats for a team using the LeagueDashTeamStats endpoint.
    """
    try:
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            measure_type_detailed_defense='Defense'
        )
        df = team_stats.get_data_frames()[0]
        team_df = df[df['TEAM_ID'] == team_id]
        if team_df.empty:
            print(f"No defensive stats found for team ID {team_id} in season {season}, {season_type}")
        return team_df
    except Exception as e:
        print(f"Error fetching defensive stats for team ID {team_id}: {e}")
        return pd.DataFrame()

def get_offensive_team_stats(team_id, season='2023-24', season_type='Regular Season'):
    """
    Fetches the offensive stats for a team using the LeagueDashTeamStats endpoint.
    """
    try:
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            measure_type_detailed_defense='Base'
        )
        df = team_stats.get_data_frames()[0]
        team_df = df[df['TEAM_ID'] == team_id]
        if team_df.empty:
            print(f"No offensive stats found for team ID {team_id} in season {season}, {season_type}")
        return team_df
    except Exception as e:
        print(f"Error fetching offensive stats for team ID {team_id}: {e}")
        return pd.DataFrame()

def get_league_defensive_averages(season='2023-24', season_type='Regular Season'):
    """
    Fetches the league-wide defensive averages for the given season.
    """
    try:
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            measure_type_detailed_defense='Defense'
        )
        df = team_stats.get_data_frames()[0]
        if df.empty:
            print(f"No league defensive stats found for season {season}, {season_type}")
            return pd.Series()
        return df.mean(numeric_only=True)
    except Exception as e:
        print(f"Error fetching league defensive averages: {e}")
        return pd.Series()

def get_player_season_recent_averages(player_id, season='2023-24', season_type='Regular Season'):
    """
    Fetches the player's season and recent (last 10 games) averages.
    """
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season, season_type_all_star=season_type)
        df = gamelog.get_data_frames()[0]
        if df.empty:
            print(f"No game logs found for player ID {player_id} in season {season}, {season_type}")
            return {'season_averages': pd.Series(), 'recent_averages': pd.Series()}
        season_avg = df[['PTS', 'REB', 'AST', 'BLK', 'STL']].mean()
        recent_avg = df.tail(10)[['PTS', 'REB', 'AST', 'BLK', 'STL']].mean()
        return {'season_averages': season_avg, 'recent_averages': recent_avg}
    except Exception as e:
        print(f"Error fetching player averages for player ID {player_id}: {e}")
        return {'season_averages': pd.Series(), 'recent_averages': pd.Series()}
