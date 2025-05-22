import pandas as pd
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, teamgamelog, leaguedashteamstats, PlayerDashboardByGeneralSplits
import time
import json
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_player_id(player_name):
    try:
        player_dict = players.find_players_by_full_name(player_name)
        if not player_dict:
            raise ValueError(f"Player '{player_name}' not found.")
        return player_dict[0]['id']
    except Exception as e:
        raise ValueError(f"Error fetching player ID for '{player_name}': {e}")

def get_team_id(team_abbr):
    try:
        team_dict = teams.find_team_by_abbreviation(team_abbr.upper())
        if not team_dict:
            raise ValueError(f"Team '{team_abbr}' not found.")
        return team_dict['id']
    except Exception as e:
        raise ValueError(f"Error fetching team ID for '{team_abbr}': {e}")

def get_head_to_head_stats(player_id, opponent_abbr, seasons=['2023-24', '2024-25']):
    game_logs = []
    h2h_games = []
    for season in seasons:
        for season_type in ['Regular Season', 'Playoffs']:
            try:
                gamelog = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=season,
                    season_type_all_star=season_type,
                    timeout=60
                )
                df = gamelog.get_data_frames()[0]
                if df.empty:
                    continue
                df['SEASON'] = season
                df['SEASON_TYPE'] = season_type
                df['OPPONENT'] = df['MATCHUP'].apply(lambda x: x.split(' vs. ')[1] if ' vs. ' in x else x.split(' @ ')[1])
                head_to_head = df[df['OPPONENT'] == opponent_abbr]
                for _, game in head_to_head.iterrows():
                    h2h_games.append({
                        'Game_Date': game['GAME_DATE'],
                        'Matchup': game['MATCHUP'],
                        'PTS': game['PTS'],
                        'REB': game['REB'],
                        'AST': game['AST'],
                        'BLK': game['BLK'],
                        'STL': game['STL']
                    })
                game_logs.append(head_to_head)
                time.sleep(0.5)
            except json.decoder.JSONDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Error fetching H2H stats for season {season}, type {season_type}: {e}")
                continue
    if game_logs:
        df = pd.concat(game_logs).reset_index(drop=True)
        return df, h2h_games
    return pd.DataFrame(), []

def get_team_recent_stats(team_id, season='2024-25', season_type='Regular Season', measure_type='Defense', num_games=10):
    stats = {
        'TEAM_ID': team_id,
        'PTS': 112.0 if measure_type == 'Base' else 110.0,
        'REB': 43.0,
        'AST': 25.0,
        'BLK': 5.0,
        'STL': 7.0,
        'PACE': 100.0,
        'WIN_PCT': 0.5,
        'OFF_RATING': None,
        'DEF_RATING': None
    }
    try:
        # Fetch team game logs
        gamelog = teamgamelog.TeamGameLog(
            team_id=team_id,
            season=season,
            season_type_all_star=season_type,
            timeout=60
        )
        df = gamelog.get_data_frames()[0]
        if df.empty:
            logger.warning("Empty game log data")
            return pd.Series(stats)
        
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='mixed').dt.date
        df = df.sort_values('GAME_DATE', ascending=False).head(num_games)
        
        # Fetch possession-related stats
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            team_id_nullable=team_id,
            season=season,
            season_type_all_star=season_type,
            measure_type_detailed_defense='Base',
            last_n_games=num_games,
            timeout=60
        )
        team_df = team_stats.get_data_frames()[0]
        fga = team_df['FGA'].iloc[0] if not team_df.empty and 'FGA' in team_df else 85.0
        orb = team_df['OREB'].iloc[0] if not team_df.empty and 'OREB' in team_df else 10.0
        tov = team_df['TOV'].iloc[0] if not team_df.empty and 'TOV' in team_df else 14.0
        fta = team_df['FTA'].iloc[0] if not team_df.empty and 'FTA' in team_df else 22.0
        pace = team_df['PACE'].iloc[0] if not team_df.empty and 'PACE' in team_df else 100.0
        
        # Calculate possessions with validation
        possessions = fga - orb + tov + (0.44 * fta)  # Updated to 0.44 per recent standards[](https://squared2020.com/2017/11/05/defensive-ratings-estimation-vs-counting/)
        if possessions <= 0 or possessions > 200:  # Typical game possessions ~80-120
            logger.warning(f"Invalid possessions: {possessions}, using default 100.0")
            possessions = 100.0
        
        # Compute averages
        stats.update({
            'PTS': min(df['PTS'].mean(), 130.0 if measure_type == 'Base' else 120.0),
            'REB': min(df['REB'].mean(), 55.0),
            'AST': min(df['AST'].mean(), 35.0),
            'BLK': min(df['BLK'].mean(), 8.0),
            'STL': min(df['STL'].mean(), 10.0),
            'PACE': pace,
            'WIN_PCT': df['WL'].apply(lambda x: 1 if x == 'W' else 0).mean()
        })
        
        # Calculate ratings
        points = stats['PTS']
        if measure_type == 'Base':
            off_rating = df['OFF_RATING'].mean() if 'OFF_RATING' in df and not df['OFF_RATING'].isna().all() else (points / possessions) * 100
            stats['OFF_RATING'] = min(off_rating, 125.0)
        else:
            def_rating = df['DEF_RATING'].mean() if 'DEF_RATING' in df and not df['DEF_RATING'].isna().all() else (points / possessions) * 100
            # Validate Defensive Rating
            if def_rating < 50.0 or def_rating > 150.0:
                logger.warning(f"Implausible DEF_RATING: {def_rating}, using default 110.0")
                def_rating = 110.0
            stats['DEF_RATING'] = min(def_rating, 125.0)
        
        logger.debug(f"Team stats: fga={fga}, orb={orb}, tov={tov}, fta={fta}, possessions={possessions}, DEF_RATING={stats['DEF_RATING']}")
        time.sleep(0.5)
    except Exception as e:
        logger.error(f"Error in get_team_recent_stats: {e}")
    
    return pd.Series(stats)

def get_defensive_team_stats(team_id, season='2024-25', season_type='Regular Season'):
    return get_team_recent_stats(team_id, season, season_type, measure_type='Defense')

def get_offensive_team_stats(team_id, season='2024-25', season_type='Regular Season'):
    return get_team_recent_stats(team_id, season, season_type, measure_type='Base')

def get_player_advanced_stats(player_id, season='2024-25', season_type='Regular Season'):
    try:
        advanced = PlayerDashboardByGeneralSplits(
            player_id=player_id,
            season=season,
            season_type_playoffs=season_type,
            measure_type_detailed='Advanced',
            timeout=60
        )
        df = advanced.get_data_frames()[0]
        return pd.Series({
            'USG_PCT': df['USG_PCT'].iloc[0] if not df.empty and 'USG_PCT' in df else 0.2,
            'TS_PCT': df['TS_PCT'].iloc[0] if not df.empty and 'TS_PCT' in df else 0.5
        })
    except Exception:
        return pd.Series({'USG_PCT': 0.2, 'TS_PCT': 0.5})

def get_player_fatigue_metrics(player_id, season='2024-25', season_type='Regular Season', num_games=10):
    try:
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star=season_type,
            timeout=60
        )
        df = gamelog.get_data_frames()[0]
        if df.empty:
            return pd.Series({'AVG_MIN': 30.0, 'AVG_REST_DAYS': 2.0})
        recent_games = df.head(num_games).copy()
        recent_games['GAME_DATE'] = pd.to_datetime(recent_games['GAME_DATE'], format='mixed').dt.date
        recent_games = recent_games.sort_values('GAME_DATE', ascending=True)
        rest_days = recent_games['GAME_DATE'].diff().apply(lambda x: x.days if pd.notnull(x) else 2.0).mean()
        avg_min = recent_games['MIN'].mean()
        return pd.Series({'AVG_MIN': avg_min, 'AVG_REST_DAYS': rest_days})
    except Exception:
        return pd.Series({'AVG_MIN': 30.0, 'AVG_REST_DAYS': 2.0})

def get_league_defensive_averages(season='2024-25', season_type='Regular Season'):
    stats = {
        'PTS': 110.0,
        'REB': 43.0,
        'AST': 25.0,
        'BLK': 5.0,
        'STL': 7.0,
        'DEF_RATING': 110.0
    }
    try:
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            measure_type_detailed_defense='Base',
            last_n_games=100,
            timeout=60
        )
        df = team_stats.get_data_frames()[0]
        if df.empty:
            return pd.Series(stats)
        
        fga = df['FGA'].mean() if 'FGA' in df else 85.0
        orb = df['OREB'].mean() if 'OREB' in df else 10.0
        tov = df['TOV'].mean() if 'TOV' in df else 14.0
        fta = df['FTA'].mean() if 'FTA' in df else 22.0
        possessions = fga - orb + tov + (0.44 * fta)
        if possessions <= 0:
            possessions = 100.0
        
        stats.update({
            'PTS': min(df['PTS'].mean(), 120.0) if 'PTS' in df else 110.0,
            'REB': min(df['REB'].mean(), 55.0) if 'REB' in df else 43.0,
            'AST': min(df['AST'].mean(), 35.0) if 'AST' in df else 25.0,
            'BLK': min(df['BLK'].mean(), 8.0) if 'BLK' in df else 5.0,
            'STL': min(df['STL'].mean(), 10.0) if 'STL' in df else 7.0,
            'DEF_RATING': min(df['DEF_RATING'].mean(), 125.0) if 'DEF_RATING' in df and not df['DEF_RATING'].isna().all() else (stats['PTS'] / possessions) * 100
        })
    except Exception:
        pass
    
    return pd.Series(stats)

def get_player_season_recent_averages(player_id, season='2024-25', season_type='Regular Season', num_games=10):
    try:
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star=season_type,
            timeout=60
        )
        df = gamelog.get_data_frames()[0]
        if df.empty:
            return {
                'season_averages': pd.Series({'PTS': 0.0, 'REB': 0.0, 'AST': 0.0, 'BLK': 0.0, 'STL': 0.0}),
                'recent_averages': pd.Series({'PTS': 0.0, 'REB': 0.0, 'AST': 0.0, 'BLK': 0.0, 'STL': 0.0})
            }
        
        # Filter out outlier games (e.g., PTS > 50, MIN < 10)
        df = df[(df['PTS'] <= 50.0) & (df['MIN'] >= 10.0)]
        
        season_avgs = {
            'PTS': min(df['PTS'].mean(), 50.0),
            'REB': min(df['REB'].mean(), 20.0),
            'AST': min(df['AST'].mean(), 15.0),
            'BLK': min(df['BLK'].mean(), 5.0),
            'STL': min(df['STL'].mean(), 5.0)
        }
        
        recent_games = df.head(num_games)
        recent_avgs = {
            'PTS': min(recent_games['PTS'].mean(), 50.0) if not recent_games.empty else 0.0,
            'REB': min(recent_games['REB'].mean(), 20.0) if not recent_games.empty else 0.0,
            'AST': min(recent_games['AST'].mean(), 15.0) if not recent_games.empty else 0.0,
            'BLK': min(recent_games['BLK'].mean(), 5.0) if not recent_games.empty else 0.0,
            'STL': min(recent_games['STL'].mean(), 5.0) if not recent_games.empty else 0.0
        }
        
        return {
            'season_averages': pd.Series(season_avgs),
            'recent_averages': pd.Series(recent_avgs)
        }
    except Exception:
        return {
            'season_averages': pd.Series({'PTS': 0.0, 'REB': 0.0, 'AST': 0.0, 'BLK': 0.0, 'STL': 0.0}),
            'recent_averages': pd.Series({'PTS': 0.0, 'REB': 0.0, 'AST': 0.0, 'BLK': 0.0, 'STL': 0.0})
        }
