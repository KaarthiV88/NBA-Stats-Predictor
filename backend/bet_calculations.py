import pandas as pd
from nba_api.stats.endpoints import playergamelog, leaguedashplayerstats, leaguedashteamstats, teamgamelog
from nba_api.stats.static import teams, players
from joblib import Memory
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for API calls
cache = Memory(location='./bet_cache', verbose=0)

def get_player_headshot_url(player_id):
    """Constructs the URL for a player's headshot."""
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png?imwidth=1040&imheight=760"

@cache.cache
def get_player_id(player_name):
    """Get player ID and headshot URL from full name."""
    try:
        player = players.find_players_by_full_name(player_name)
        if not player:
            logger.error(f"Player {player_name} not found")
            return None
        player_id = player[0]['id']
        headshot_url = get_player_headshot_url(player_id)
        logger.debug(f"Player ID for {player_name}: {player_id}, Headshot URL: {headshot_url}")
        return {'player_id': player_id, 'headshot_url': headshot_url}
    except Exception as e:
        logger.error(f"Error fetching player ID for {player_name}: {e}")
        return None

@cache.cache
def get_team_id(team_abbr):
    """Get team ID from abbreviation."""
    try:
        team = teams.find_team_by_abbreviation(team_abbr)
        if not team:
            logger.error(f"Team abbreviation {team_abbr} not found")
            return None
        logger.debug(f"Team ID for {team_abbr}: {team['id']}")
        return team['id']
    except Exception as e:
        logger.error(f"Error fetching team ID for {team_abbr}: {e}")
        return None

@cache.cache
def get_player_season_recent_averages(player_id, season, season_type, recent_n=10):
    """Get season and recent game averages for a player."""
    try:
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star=season_type,
            timeout=60
        ).get_data_frames()[0]
        if gamelog.empty:
            logger.warning(f"No game log data for player {player_id}, season {season}")
            return {
                'season_averages': {'PTS': 0.0, 'REB': 0.0, 'AST': 0.0, 'BLK': 0.0, 'STL': 0.0},
                'recent_averages': {'PTS': 0.0, 'REB': 0.0, 'AST': 0.0, 'BLK': 0.0, 'STL': 0.0}
            }
        season_avgs = gamelog[['PTS', 'REB', 'AST', 'BLK', 'STL']].mean().to_dict()
        recent_games = gamelog.head(recent_n)
        recent_avgs = recent_games[['PTS', 'REB', 'AST', 'BLK', 'STL']].mean().to_dict()
        return {
            'season_averages': {k: v for k, v in season_avgs.items() if not pd.isna(v)},
            'recent_averages': {k: v for k, v in recent_avgs.items() if not pd.isna(v)}
        }
    except Exception as e:
        logger.error(f"Error fetching averages for player {player_id}, season {season}: {e}")
        return {
            'season_averages': {'PTS': 0.0, 'REB': 0.0, 'AST': 0.0, 'BLK': 0.0, 'STL': 0.0},
            'recent_averages': {'PTS': 0.0, 'REB': 0.0, 'AST': 0.0, 'BLK': 0.0, 'STL': 0.0}
        }

@cache.cache
def get_head_to_head_stats(player_id, opponent_abbr, seasons=('2023-24', '2024-25')):
    """Get head-to-head stats vs. an opponent."""
    opponent_id = get_team_id(opponent_abbr)
    if not opponent_id:
        logger.warning(f"No opponent ID for {opponent_abbr}")
        return pd.DataFrame(), []
    game_logs = []
    for season in seasons:
        try:
            gl = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star='Regular Season',
                timeout=60
            ).get_data_frames()[0]
            gl['OPPONENT'] = gl['MATCHUP'].apply(
                lambda x: x.split(' vs. ')[1] if ' vs. ' in x else x.split(' @ ')[1]
            )
            game_logs.append(gl)
        except Exception as e:
            logger.warning(f"Error fetching H2H for player {player_id}, season {season}: {e}")
    if not game_logs:
        return pd.DataFrame(), []
    all_games = pd.concat(game_logs).reset_index(drop=True)
    h2h_games = all_games[all_games['OPPONENT'] == opponent_abbr]
    h2h_stats = h2h_games[['PTS', 'REB', 'AST', 'BLK', 'STL']]
    h2h_list = [
        {
            'Game_Date': row['GAME_DATE'],
            'Matchup': row['MATCHUP'],
            'PTS': float(row['PTS']),
            'REB': float(row['REB']),
            'AST': float(row['AST']),
            'BLK': float(row['BLK']),
            'STL': float(row['STL'])
        }
        for _, row in h2h_games.iterrows()
    ]
    logger.debug(f"H2H games for {player_id} vs {opponent_abbr}: {len(h2h_list)}")
    return h2h_stats, h2h_list

@cache.cache
def get_league_defensive_averages(season, season_type):
    """Get league-wide defensive averages."""
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            measure_type_detailed_defense='Defense',
            timeout=60
        ).get_data_frames()[0]
        logger.debug(f"Available columns in team stats for {season}: {stats.columns.tolist()}")
        # Map expected stats to API defensive stat columns
        column_mapping = {
            'PTS': 'OPP_PTS',  # Opponent points allowed
            'REB': 'DREB',     # Defensive rebounds
            'AST': 'OPP_AST',  # Opponent assists allowed
            'BLK': 'BLK',      # Blocks (assumed correct, no warning)
            'STL': 'STL'       # Steals (assumed correct, no warning)
        }
        avgs = {}
        for stat, api_col in column_mapping.items():
            if api_col in stats.columns:
                avgs[stat] = stats[api_col].mean()
            else:
                logger.warning(f"Column {api_col} not found for {season}, using default")
                avgs[stat] = {'PTS': 110.0, 'REB': 43.0, 'AST': 25.0, 'BLK': 5.0, 'STL': 7.0}.get(stat)
        logger.debug(f"League averages for {season}: {avgs}")
        return avgs
    except Exception as e:
        logger.error(f"Error fetching league averages for {season}: {e}")
        return {'PTS': 110.0, 'REB': 43.0, 'AST': 25.0, 'BLK': 5.0, 'STL': 7.0}

@cache.cache
def get_team_recent_stats(team_id, season, season_type, measure_type, num_games=10):
    """Get recent team stats."""
    try:
        gamelog = teamgamelog.TeamGameLog(
            team_id=team_id,
            season=season,
            season_type_all_star=season_type,
            timeout=60
        ).get_data_frames()[0]
        recent = gamelog.head(num_games)
        stats = recent[['PTS', 'REB', 'AST', 'BLK', 'STL']].mean().to_dict()
        stats['PACE'] = recent.get('PACE', 100.0).mean() if 'PACE' in recent else 100.0
        stats['DEF_RATING'] = recent.get('DEF_RATING', 110.0).mean() if measure_type == 'Defense' and 'DEF_RATING' in recent else 110.0
        stats['OFF_RATING'] = recent.get('OFF_RATING', 110.0).mean() if measure_type == 'Base' and 'OFF_RATING' in recent else 110.0
        logger.debug(f"Team stats for team {team_id}, season {season}: {stats}")
        return stats
    except Exception as e:
        logger.error(f"Error fetching team stats for {team_id}, season {season}: {e}")
        return {
            'PTS': 110.0,
            'REB': 43.0,
            'AST': 25.0,
            'BLK': 5.0,
            'STL': 7.0,
            'PACE': 100.0,
            'DEF_RATING': 110.0,
            'OFF_RATING': 110.0
        }

@cache.cache
def get_player_advanced_stats(player_id, season, season_type):
    """Get advanced player stats."""
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star=season_type,
            timeout=60
        ).get_data_frames()[0]
        player_stats = stats[stats['PLAYER_ID'] == player_id]
        if player_stats.empty:
            logger.warning(f"No advanced stats for player {player_id}, season {season}")
            return {'USG_PCT': 0.2, 'TS_PCT': 0.5}
        player_stats = player_stats.iloc[0]
        return {
            'USG_PCT': player_stats.get('USG_PCT', 0.2),
            'TS_PCT': player_stats.get('TS_PCT', 0.5)
        }
    except Exception as e:
        logger.error(f"Error fetching advanced stats for {player_id}, season {season}: {e}")
        return {'USG_PCT': 0.2, 'TS_PCT': 0.5}

@cache.cache
def get_player_fatigue_metrics(player_id, season, season_type, num_games=10):
    """Get fatigue metrics for a player."""
    try:
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star=season_type,
            timeout=60
        ).get_data_frames()[0]
        recent = gamelog.head(num_games).copy()  # Create a copy to avoid SettingWithCopyWarning
        avg_min = recent['MIN'].mean() if not recent.empty else 30.0
        recent['GAME_DATE'] = pd.to_datetime(recent['GAME_DATE'], format='mixed', errors='coerce')
        rest_days = recent['GAME_DATE'].diff().dt.days.dropna().mean() if not recent.empty and len(recent) > 1 else 2.0
        logger.debug(f"Fatigue metrics for {player_id}: AVG_MIN={avg_min}, AVG_REST_DAYS={rest_days}")
        return {'AVG_MIN': avg_min, 'AVG_REST_DAYS': rest_days}
    except Exception as e:
        logger.error(f"Error fetching fatigue metrics for {player_id}: {e}")
        return {'AVG_MIN': 30.0, 'AVG_REST_DAYS': 2.0}
