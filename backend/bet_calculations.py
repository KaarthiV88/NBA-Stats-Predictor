import pandas as pd
from nba_api.stats.endpoints import playergamelog, leaguedashplayerstats, leaguedashteamstats, teamgamelog, commonplayerinfo
from nba_api.stats.static import teams, players
from joblib import Memory
import logging
import os
from retrying import retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for API calls
cache = Memory(location='./bet_cache', verbose=0)
cache.clear()  # Clear cache on startup to ensure fresh data

# NBA Team Primary Colors (official team colors)
TEAM_COLORS = {
    'ATL': '#E03A3E',  # Atlanta Hawks - Red
    'BOS': '#007A33',  # Boston Celtics - Green
    'BKN': '#ADACB5',  # Brooklyn Nets - Silver
    'CHA': '#00788C',  # Charlotte Hornets - Teal
    'CHI': '#CE1141',  # Chicago Bulls - Red
    'CLE': '#860038',  # Cleveland Cavaliers - Wine
    'DAL': '#00538C',  # Dallas Mavericks - Blue
    'DEN': '#FEC524',  # Denver Nuggets - Yellow
    'DET': '#C8102E',  # Detroit Pistons - Red
    'GSW': '#1D428A',  # Golden State Warriors - Blue
    'HOU': '#CE1141',  # Houston Rockets - Red
    'IND': '#fdbb30',  # Indiana Pacers - Yellow
    'LAC': '#C8102E',  # LA Clippers - Red
    'LAL': '#8A56B9',  # Los Angeles Lakers - Purple
    'MEM': '#5D76A9',  # Memphis Grizzlies - Blue
    'MIA': '#98002E',  # Miami Heat - Red
    'MIL': '#00471B',  # Milwaukee Bucks - Green
    'MIN': '#2E4E7E',  # Minnesota Timberwolves - Blue
    'NOP': '#2E4E7E',  # New Orleans Pelicans - Blue
    'NYK': '#E6731B',  # New York Knicks - Orange
    'OKC': '#007AC1',  # Oklahoma City Thunder - Blue
    'ORL': '#0077C0',  # Orlando Magic - Blue
    'PHI': '#006BB6',  # Philadelphia 76ers - Blue/Red
    'PHX': '#E56020',  # Phoenix Suns - Orange
    'POR': '#E03A3E',  # Portland Trail Blazers - Red
    'SAC': '#5A2D81',  # Sacramento Kings - Purple
    'SAS': '#C4CED4',  # San Antonio Spurs - Silver
    'TOR': '#CE1141',  # Toronto Raptors - Red
    'UTA': '#753BBD',  # Utah Jazz - Purple
    'WAS': '#002B5C'   # Washington Wizards - Blue
}

def get_team_primary_color(team_abbr):
    """Get the primary color for a team by abbreviation."""
    return TEAM_COLORS.get(team_abbr.upper(), '#3EB489')  # Default to mint green if team not found

def get_player_headshot_url(player_id):
    """Constructs the URL for a player's headshot."""
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png?imwidth=1040&imheight=760"

@cache.cache
def get_player_detailed_info(player_id):
    """Get detailed player information including height, weight, jersey, position, team, school, country."""
    try:
        player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
        if player_info.empty:
            logger.error(f"No detailed info found for player {player_id}")
            return None
        
        player_data = player_info.iloc[0]
        team_abbr = player_data.get('TEAM_ABBREVIATION', 'N/A')
        team_color = get_team_primary_color(team_abbr) if team_abbr != 'N/A' else '#3EB489'
        
        return {
            'height': str(player_data.get('HEIGHT', 'N/A')),
            'weight': str(player_data.get('WEIGHT', 'N/A')),
            'jersey': str(player_data.get('JERSEY', 'N/A')),
            'position': str(player_data.get('POSITION', 'N/A')),
            'team_name': str(player_data.get('TEAM_NAME', 'N/A')),
            'team_city': str(player_data.get('TEAM_CITY', 'N/A')),
            'team_abbreviation': str(player_data.get('TEAM_ABBREVIATION', 'N/A')),
            'team_color': team_color,
            'school': str(player_data.get('SCHOOL', 'N/A')),
            'country': str(player_data.get('COUNTRY', 'N/A')),
            'season_exp': int(player_data.get('SEASON_EXP', 0)) if pd.notna(player_data.get('SEASON_EXP')) else 'N/A'
        }
    except Exception as e:
        logger.error(f"Error fetching detailed player info for {player_id}: {e}")
        return None

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

@retry(stop_max_attempt_number=3, wait_fixed=2000)
def fetch_game_log(player_id, season, season_type):
    """Fetch game log with retries."""
    return playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star=season_type,
        timeout=120
    ).get_data_frames()[0]

@retry(stop_max_attempt_number=3, wait_fixed=2000)
def fetch_advanced_stats(player_id, season, season_type):
    """Fetch advanced stats with retries."""
    all_stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star=season_type,
        timeout=120
    ).get_data_frames()[0]
    # Filter for the specific player
    player_stats = all_stats[all_stats['PLAYER_ID'] == player_id]
    return player_stats

@cache.cache
def get_player_season_recent_averages(player_id, season, season_type, recent_n=10):
    """Get season and recent game averages for a player across all stats."""
    logger.info(f"Fetching averages for player_id: {player_id}, season: {season}, season_type: {season_type}")
    try:
        # Prioritize 2023-24 for stability, fallback to 2024-25 and 2022-23
        seasons_to_try = ['2023-24', '2024-25', '2022-23']
        gamelog = None
        for s in seasons_to_try:
            try:
                gamelog = fetch_game_log(player_id, s, season_type)
                logger.info(f"Game log for player {player_id}, season {s}: {gamelog.shape[0]} games")
                if not gamelog.empty:
                    logger.info(f"Successfully fetched game log for player {player_id}, season {s}")
                    break
                logger.warning(f"Empty game log for player {player_id}, season {s}")
            except Exception as e:
                logger.warning(f"Error fetching game log for player {player_id}, season {s}: {e}")
        if gamelog is None or gamelog.empty:
            logger.error(f"No valid game log data for player {player_id} across seasons {seasons_to_try}")
            return {
                'season_averages': {'PTS': 0.0, 'REB': 0.0, 'AST': 0.0, 'BLK': 0.0, 'STL': 0.0, 'DEF_RATING': 110.0},
                'recent_averages': {'PTS': 0.0, 'REB': 0.0, 'AST': 0.0, 'BLK': 0.0, 'STL': 0.0, 'DEF_RATING': 110.0}
            }

        # Calculate basic averages
        season_avgs = gamelog[['PTS', 'REB', 'AST', 'BLK', 'STL']].mean().to_dict()
        recent_games = gamelog.head(recent_n)
        recent_avgs = recent_games[['PTS', 'REB', 'AST', 'BLK', 'STL']].mean().to_dict()

        # Fetch defensive rating
        try:
            advanced_stats = fetch_advanced_stats(player_id, season, season_type)
            logger.info(f"Advanced stats for player {player_id}, season {season}: {advanced_stats.shape[0]} rows")
            def_rating = advanced_stats['DEF_RATING'].iloc[0] if not advanced_stats.empty else 110.0
            if def_rating == 0.0:
                logger.warning(f"No DEF_RATING found for player {player_id}, using league average 110.0")
                def_rating = 110.0
        except Exception as e:
            logger.warning(f"Error fetching advanced stats for player {player_id}, season {season}: {e}")
            def_rating = 110.0

        season_avgs['DEF_RATING'] = float(def_rating)
        recent_avgs['DEF_RATING'] = float(def_rating)

        return {
            'season_averages': {k: float(v) for k, v in season_avgs.items() if not pd.isna(v)},
            'recent_averages': {k: float(v) for k, v in recent_avgs.items() if not pd.isna(v)}
        }
    except Exception as e:
        logger.error(f"Unexpected error fetching averages for player {player_id}, season {season}: {e}")
        return {
            'season_averages': {'PTS': 0.0, 'REB': 0.0, 'AST': 0.0, 'BLK': 0.0, 'STL': 0.0, 'DEF_RATING': 110.0},
            'recent_averages': {'PTS': 0.0, 'REB': 0.0, 'AST': 0.0, 'BLK': 0.0, 'STL': 0.0, 'DEF_RATING': 110.0}
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
            gl = fetch_game_log(player_id, season, 'Regular Season')
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
            timeout=120
        ).get_data_frames()[0]
        logger.info(f"Available columns in team stats for {season}: {stats.columns.tolist()}")
        
        # Try different possible column names for each stat
        column_mapping = {
            'PTS': ['OPP_PTS', 'OPP_PTS_PG', 'OPP_POINTS', 'OPP_PTS_PER_GAME'],
            'REB': ['DREB', 'OPP_REB', 'OPP_REB_PG', 'OPP_REB_PER_GAME'],
            'AST': ['OPP_AST', 'OPP_AST_PG', 'OPP_ASSISTS', 'OPP_AST_PER_GAME'],
            'BLK': ['BLK', 'BLK_PG', 'BLOCKS', 'BLK_PER_GAME'],
            'STL': ['STL', 'STL_PG', 'STEALS', 'STL_PER_GAME']
        }
        
        avgs = {}
        for stat, possible_cols in column_mapping.items():
            found_col = None
            for col in possible_cols:
                if col in stats.columns:
                    found_col = col
                    break
            
            if found_col:
                avgs[stat] = stats[found_col].mean()
                logger.debug(f"Using column '{found_col}' for {stat}")
            else:
                logger.warning(f"No column found for {stat} in {season}, using default. Available columns: {[col for col in stats.columns if stat.lower() in col.lower() or 'opp' in col.lower()]}")
                avgs[stat] = {'PTS': 110.0, 'REB': 43.0, 'AST': 25.0, 'BLK': 5.0, 'STL': 7.0}.get(stat)
        
        logger.debug(f"League averages for {season}: {avgs}")
        return avgs
    except Exception as e:
        logger.error(f"Error fetching league averages for {season}: {e}")
        return {'PTS': 110.0, 'REB': 43.0, 'AST': 25.0, 'BLK': 5.0, 'STL': 7.0}

@cache.cache
def get_team_recent_stats(team_id, season, season_type, measure_type, num_games=10):
    """Get recent team stats (per-game averages for last num_games)."""
    try:
        gamelog = teamgamelog.TeamGameLog(
            team_id=team_id,
            season=season,
            season_type_all_star=season_type,
            timeout=120
        ).get_data_frames()[0]
        if gamelog.empty:
            logger.warning(f"No game log data for team {team_id}, season {season}")
            raise ValueError("Empty game log")
        recent = gamelog.head(num_games)
        if recent.empty:
            logger.warning(f"No recent games for team {team_id}, season {season}")
            raise ValueError("No recent games")
        # Calculate per-game averages
        stats = {}
        for stat in ['PTS', 'REB', 'AST', 'BLK', 'STL']:
            stats[stat] = recent[stat].mean() if stat in recent else None
        stats['PACE'] = recent['PACE'].mean() if 'PACE' in recent else None
        stats['DEF_RATING'] = recent['DEF_RATING'].mean() if 'DEF_RATING' in recent else None
        stats['OFF_RATING'] = recent['OFF_RATING'].mean() if 'OFF_RATING' in recent else None
        # Fallback to defaults if any stat is missing
        for k, v in stats.items():
            if v is None or pd.isna(v):
                logger.warning(f"Missing {k} for team {team_id}, using default value.")
                stats[k] = {'PTS': 110.0, 'REB': 43.0, 'AST': 25.0, 'BLK': 5.0, 'STL': 7.0, 'PACE': 100.0, 'DEF_RATING': 110.0, 'OFF_RATING': 110.0}[k]
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
        stats = fetch_advanced_stats(player_id, season, season_type)
        if stats.empty:
            logger.warning(f"No advanced stats for player {player_id}, season {season}")
            return {'USG_PCT': 0.2, 'TS_PCT': 0.5, 'DEF_RATING': 110.0}
        player_stats = stats.iloc[0]
        return {
            'USG_PCT': player_stats.get('USG_PCT', 0.2),
            'TS_PCT': player_stats.get('TS_PCT', 0.5),
            'DEF_RATING': player_stats.get('DEF_RATING', 110.0)
        }
    except Exception as e:
        logger.error(f"Error fetching advanced stats for {player_id}, season {season}: {e}")
        return {'USG_PCT': 0.2, 'TS_PCT': 0.5, 'DEF_RATING': 110.0}

@cache.cache
def get_player_fatigue_metrics(player_id, season, season_type, num_games=10):
    """Get fatigue metrics for a player."""
    try:
        gamelog = fetch_game_log(player_id, season, season_type)
        recent = gamelog.head(num_games).copy()
        avg_min = recent['MIN'].mean() if not recent.empty else 30.0
        recent['GAME_DATE'] = pd.to_datetime(recent['GAME_DATE'], format='mixed', errors='coerce')
        rest_days = recent['GAME_DATE'].diff().dt.days.dropna().mean() if not recent.empty and len(recent) > 1 else 2.0
        logger.debug(f"Fatigue metrics for {player_id}: AVG_MIN={avg_min}, AVG_REST_DAYS={rest_days}")
        return {'AVG_MIN': avg_min, 'AVG_REST_DAYS': rest_days}
    except Exception as e:
        logger.error(f"Error fetching fatigue metrics for {player_id}: {e}")
        return {'AVG_MIN': 30.0, 'AVG_REST_DAYS': 2.0}

def get_all_active_players():
    """Fetch a list of all current active NBA players from nba_api."""
    try:
        active_players = players.get_active_players()
        if not active_players:
            logger.warning("No active players retrieved from nba_api")
            return []
        player_list = [{'id': player['id'], 'full_name': player['full_name']} for player in active_players]
        logger.info(f"Retrieved {len(player_list)} active players")
        return player_list
    except Exception as e:
        logger.error(f"Error fetching active players: {str(e)}")
        return []
