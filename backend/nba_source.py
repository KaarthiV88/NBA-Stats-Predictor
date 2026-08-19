"""Single access point for every piece of NBA data the app reads.

Two modes, selected by the NBA_SNAPSHOT environment variable:

  live (default)      Calls stats.nba.com, exactly as the app always has.
                      Used for local development and for building snapshots.

  snapshot (=1)       Reads committed JSON under ./data. Makes zero network
                      calls. Used in production.

Why snapshot mode exists
------------------------
stats.nba.com silently drops traffic originating from datacenter IP ranges
(AWS/GCP/Azure) -- requests hang rather than returning an error. Any host we
could deploy to for free sits in one of those ranges, so the deployed backend
simply cannot reach it. Since the underlying data changes at most once a day
and is identical for every visitor, the fix is to fetch it somewhere the block
doesn't apply (a residential connection, via build_snapshot.py) and ship the
result alongside the code.

Every function here returns a pandas DataFrame with the same shape the nba_api
endpoint returns, so callers cannot tell the two modes apart. On a snapshot
miss the functions return an empty DataFrame rather than raising: every caller
already has a defaults path for empty results, so a gap degrades the prediction
instead of failing the request.
"""

import json
import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# --- Mode ----------------------------------------------------------------

SNAPSHOT_MODE = os.environ.get('NBA_SNAPSHOT', '').strip().lower() in ('1', 'true', 'yes')

# Resolved against this file, not the working directory -- hosts invoke the app
# from the project root, which is not necessarily this package's directory.
SNAPSHOT_DIR = Path(os.environ.get('NBA_SNAPSHOT_DIR') or (Path(__file__).parent / 'data'))

# Game-log columns the model and the API responses actually consume. Storing
# only these keeps the whole 530-player snapshot near 6 MB instead of ~20 MB.
GAMELOG_COLUMNS = ['GAME_DATE', 'MATCHUP', 'MIN', 'PTS', 'REB', 'AST', 'BLK', 'STL']

# Per-request timeout. The app wants a generous value; bulk builds want a short
# one, because a throttled request that blocks for two minutes stalls the whole
# run. build_snapshot.py lowers this via NBA_REQUEST_TIMEOUT.
_REQUEST_TIMEOUT = int(os.environ.get('NBA_REQUEST_TIMEOUT', '120'))


def _slug(value):
    """Filename-safe form of a season or season type."""
    return str(value).strip().lower().replace(' ', '-').replace('/', '-')


def _read_json(path):
    """Return parsed JSON at path, or None when it isn't there."""
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Unreadable snapshot file %s: %s", path, exc)
        return None


def _frame(records, columns=None):
    """Build a DataFrame, preserving column layout even when there are no rows."""
    if records:
        return pd.DataFrame.from_records(records)
    return pd.DataFrame(columns=pd.Index(columns or []))


# --- Snapshot paths ------------------------------------------------------

def gamelog_path(player_id):
    return SNAPSHOT_DIR / 'gamelogs' / f'{player_id}.json'


def team_gamelog_path(team_id):
    return SNAPSHOT_DIR / 'team_gamelogs' / f'{team_id}.json'


def player_info_path():
    return SNAPSHOT_DIR / 'player_info.json'


def league_player_stats_path(season, season_type):
    return SNAPSHOT_DIR / 'league_player_stats' / f'{_slug(season)}__{_slug(season_type)}.json'


def league_team_stats_path(season, season_type, last_n_games):
    tail = _slug(last_n_games) if last_n_games else 'all'
    return SNAPSHOT_DIR / 'league_team_stats' / f'{_slug(season)}__{_slug(season_type)}__{tail}.json'


def league_opponent_stats_path(season, season_type):
    return SNAPSHOT_DIR / 'league_opponent_stats' / f'{_slug(season)}__{_slug(season_type)}.json'


def season_key(season, season_type):
    """Key used inside per-player and per-team game-log files."""
    return f'{season}|{season_type}'


# --- Player game logs ----------------------------------------------------

def player_game_log(player_id, season, season_type):
    """Game log for one player, one season, one season type."""
    if SNAPSHOT_MODE:
        blob = _read_json(gamelog_path(player_id))
        if blob is None:
            logger.warning("No snapshot game log for player %s", player_id)
            return _frame([], GAMELOG_COLUMNS)
        return _frame(blob.get(season_key(season, season_type), []), GAMELOG_COLUMNS)

    from nba_api.stats.endpoints import playergamelog
    return playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star=season_type,
        timeout=_REQUEST_TIMEOUT,
    ).get_data_frames()[0]


# --- Player biography ----------------------------------------------------

def player_common_info(player_id):
    """CommonPlayerInfo row for one player, as a single-row DataFrame."""
    if SNAPSHOT_MODE:
        blob = _read_json(player_info_path()) or {}
        record = blob.get(str(player_id))
        if record is None:
            logger.warning("No snapshot bio for player %s", player_id)
            return _frame([])
        return _frame([record])

    from nba_api.stats.endpoints import commonplayerinfo
    return commonplayerinfo.CommonPlayerInfo(
        player_id=player_id,
        timeout=_REQUEST_TIMEOUT,
    ).get_data_frames()[0]


# --- League-wide player stats -------------------------------------------

def league_player_stats(season, season_type):
    """LeagueDashPlayerStats for a season -- one row per player."""
    if SNAPSHOT_MODE:
        return _frame(_read_json(league_player_stats_path(season, season_type)) or [])

    from nba_api.stats.endpoints import leaguedashplayerstats
    return leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star=season_type,
        timeout=_REQUEST_TIMEOUT,
    ).get_data_frames()[0]


# --- League-wide team stats ---------------------------------------------

def league_team_stats(season, season_type, last_n_games=None):
    """LeagueDashTeamStats (Defense) for a season -- one row per team.

    last_n_games mirrors the API parameter; callers pass '82' for a full-season
    window and None for the endpoint's default.
    """
    if SNAPSHOT_MODE:
        return _frame(_read_json(league_team_stats_path(season, season_type, last_n_games)) or [])

    from nba_api.stats.endpoints import leaguedashteamstats
    kwargs = {
        'season': season,
        'season_type_all_star': season_type,
        'measure_type_detailed_defense': 'Defense',
        'timeout': _REQUEST_TIMEOUT,
    }
    if last_n_games is not None:
        kwargs['last_n_games'] = last_n_games
    return leaguedashteamstats.LeagueDashTeamStats(**kwargs).get_data_frames()[0]


def league_opponent_stats(season, season_type):
    """LeagueDashTeamStats with the Opponent measure -- what each team allows.

    The Defense measure does not carry opponent points or assists, only the
    defending team's own counting stats. This is the table that has OPP_PTS and
    OPP_AST, so league baselines for those two come from here.
    """
    if SNAPSHOT_MODE:
        return _frame(_read_json(league_opponent_stats_path(season, season_type)) or [])

    from nba_api.stats.endpoints import leaguedashteamstats
    return leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star=season_type,
        measure_type_detailed_defense='Opponent',
        timeout=_REQUEST_TIMEOUT,
    ).get_data_frames()[0]


# --- Team game logs ------------------------------------------------------

def team_game_log(team_id, season, season_type):
    """Game log for one team, one season, one season type."""
    if SNAPSHOT_MODE:
        blob = _read_json(team_gamelog_path(team_id))
        if blob is None:
            logger.warning("No snapshot game log for team %s", team_id)
            return _frame([])
        return _frame(blob.get(season_key(season, season_type), []))

    from nba_api.stats.endpoints import teamgamelog
    return teamgamelog.TeamGameLog(
        team_id=team_id,
        season=season,
        season_type_all_star=season_type,
        timeout=_REQUEST_TIMEOUT,
    ).get_data_frames()[0]


# --- Introspection -------------------------------------------------------

def describe():
    """Human-readable mode summary, logged at startup."""
    if not SNAPSHOT_MODE:
        return 'live (stats.nba.com)'
    meta = _read_json(SNAPSHOT_DIR / 'meta.json') or {}
    built = meta.get('built_at', 'unknown time')
    players = meta.get('player_count', '?')
    return f'snapshot ({SNAPSHOT_DIR}, {players} players, built {built})'
