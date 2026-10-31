#!/usr/bin/env python3
"""Check a freshly built snapshot before it is committed and deployed.

The scheduled refresh (scripts/refresh_snapshot.sh) runs this after
build_snapshot.py and pushes only when it passes. A failure leaves the live
site on its last good snapshot instead of shipping damaged data.

    python validate_snapshot.py            # structure + comparison with HEAD
    python validate_snapshot.py --smoke    # also run one real prediction, offline

Exit code 0 means safe to ship; 1 means do not ship, with the reasons printed.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Read the snapshot, never the network, and keep the smoke test's model cache
# out of the working tree. Both must be set before nba_source is imported.
os.environ['NBA_SNAPSHOT'] = '1'
os.environ.setdefault('BET_CACHE_DIR', tempfile.mkdtemp(prefix='kvmm-validate-'))

import nba_source  # noqa: E402
from bet_calculations import SEASONS  # noqa: E402

SEASON_TYPES = ('Regular Season', 'Playoffs')
EXPECTED_KEYS = {nba_source.season_key(s, st) for s in SEASONS for st in SEASON_TYPES}
CURRENT_REGULAR = nba_source.season_key(SEASONS[-1], 'Regular Season')
TEAM_COUNT = 30
# Allowed shrinkage in the number of player files: the active roster churns a
# little between refreshes, but a large drop means the fetch went wrong.
MIN_PLAYER_RATIO = 0.9

OUT = nba_source.SNAPSHOT_DIR
REPO = Path(subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=OUT,
                           capture_output=True, text=True, check=True).stdout.strip())

problems = []


def check(ok, message):
    if not ok:
        problems.append(message)
    return ok


def git(*args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, check=False).stdout


def rel(path):
    return path.resolve().relative_to(REPO).as_posix()


def head_json(path):
    """The committed version of a snapshot file, or None if HEAD lacks it."""
    out = git('show', f'HEAD:{rel(path)}')
    try:
        return json.loads(out) if out else None
    except ValueError:
        return None


def changed_paths():
    """Snapshot files that differ from HEAD, including new ones."""
    data = rel(OUT)
    names = git('diff', '--name-only', '-z', 'HEAD', '--', data).split(b'\0')
    names += git('ls-files', '--others', '--exclude-standard', '-z', '--', data).split(b'\0')
    return {(REPO / n.decode()).resolve() for n in names if n}


def load(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        problems.append(f'{rel(path)}: unreadable ({exc})')
        return None


def check_logs(kind, paths, changed, columns=None):
    """Season keys, row shape, and no lost games relative to HEAD."""
    bad_keys, bad_rows, shrunk = [], [], []
    for path in paths:
        blob = load(path)
        if blob is None:
            continue
        if set(blob) != EXPECTED_KEYS:
            bad_keys.append(path.stem)
            continue
        if columns and any(not set(columns) <= set(row) for rows in blob.values() for row in rows):
            bad_rows.append(path.stem)
        if path.resolve() in changed:
            old = head_json(path) or {}
            for key in EXPECTED_KEYS & set(old):
                if len(blob[key]) < len(old[key]):
                    shrunk.append(f'{path.stem} {key} {len(old[key])}->{len(blob[key])}')
    check(not bad_keys, f'{kind}: {len(bad_keys)} file(s) without exactly {sorted(EXPECTED_KEYS)}, '
                        f'e.g. {bad_keys[:5]}')
    check(not bad_rows, f'{kind}: {len(bad_rows)} file(s) with rows missing columns, e.g. {bad_rows[:5]}')
    check(not shrunk, f'{kind}: {len(shrunk)} log(s) lost games versus HEAD, e.g. {shrunk[:5]}')


def current_season_games(team_paths):
    total = 0
    for path in team_paths:
        blob = load(path) or {}
        total += len(blob.get(CURRENT_REGULAR, []))
    return total


def check_league_tables(current_games):
    for season in SEASONS:
        for season_type in SEASON_TYPES:
            tables = [
                (nba_source.league_player_stats_path(season, season_type), 200),
                (nba_source.league_opponent_stats_path(season, season_type), TEAM_COUNT),
                (nba_source.league_team_stats_path(season, season_type, None), TEAM_COUNT),
                (nba_source.league_team_stats_path(season, season_type, '82'), TEAM_COUNT),
            ]
            # Playoff tables are legitimately empty for most of the year, and the
            # new season's are empty until its first game is played.
            must_have_rows = season_type == 'Regular Season' and (season != SEASONS[-1] or current_games > 0)
            for path, min_rows in tables:
                if not check(path.exists(), f'missing league table {rel(path)}'):
                    continue
                rows = load(path)
                if not check(isinstance(rows, list), f'{rel(path)}: not a list of rows'):
                    continue
                if must_have_rows:
                    check(len(rows) >= min_rows, f'{rel(path)}: {len(rows)} rows, expected at least {min_rows}')


def check_meta():
    meta = load(OUT / 'meta.json') or {}
    check(meta.get('seasons') == list(SEASONS),
          f'meta.json seasons {meta.get("seasons")} != configured {list(SEASONS)}')
    latest = nba_source.snapshot_latest_game_date()
    check(meta.get('latest_game_date') == latest,
          f'meta.json latest_game_date {meta.get("latest_game_date")} != team logs {latest}')
    previous = (head_json(OUT / 'meta.json') or {}).get('latest_game_date')
    if previous and latest:
        check(latest >= previous, f'newest game moved backwards: {previous} -> {latest}')
    return latest


def smoke_test():
    """Train and predict for the player with the most current-season games.

    Before the new season's first game, that falls back to the player with the
    most games last season.
    """
    from predictive_model import AdvancedNBAPlayerPredictor

    previous_regular = nba_source.season_key(SEASONS[0], 'Regular Season')
    best, best_rank = None, (0, 0)
    for path in (OUT / 'gamelogs').glob('*.json'):
        blob = load(path) or {}
        current, previous = blob.get(CURRENT_REGULAR, []), blob.get(previous_regular, [])
        rank = (len(current), len(previous))
        if rank > best_rank:
            best, best_rank = (int(path.stem), current or previous), rank
    if not check(best is not None, 'smoke test: no player has any regular-season games'):
        return
    player_id, rows = best
    matchup = rows[0]['MATCHUP']
    opponent = matchup.split(' vs. ')[-1] if ' vs. ' in matchup else matchup.split(' @ ')[-1]
    line = round(sum(r['PTS'] for r in rows) / len(rows)) + 0.5
    try:
        result = AdvancedNBAPlayerPredictor().predict_over_under(
            player_id=player_id, category='POINTS', opponent_abbr=opponent,
            season_type='Regular Season', betting_line=line,
        )
    except Exception as exc:  # noqa: BLE001 - any failure here means do not ship
        problems.append(f'smoke test: prediction for player {player_id} raised {type(exc).__name__}: {exc}')
        return
    value, confidence = result.get('predicted_value'), result.get('confidence')
    check(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 80,
          f'smoke test: implausible predicted_value {value!r} for player {player_id}')
    check(isinstance(confidence, (int, float)) and 50 <= confidence <= 100,
          f'smoke test: implausible confidence {confidence!r} for player {player_id}')
    print(f'  smoke test: player {player_id} vs {opponent}, line {line} -> '
          f'{value} ({result.get("bet_on")}, {confidence:.1f}%)')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--smoke', action='store_true', help='also run one end-to-end prediction')
    args = parser.parse_args()

    changed = changed_paths()
    player_paths = sorted((OUT / 'gamelogs').glob('*.json'))
    team_paths = sorted((OUT / 'team_gamelogs').glob('*.json'))

    head_players = len([n for n in git('ls-tree', '--name-only', 'HEAD', f'{rel(OUT)}/gamelogs/').split(b'\n') if n])
    if head_players:
        check(len(player_paths) >= MIN_PLAYER_RATIO * head_players,
              f'player files dropped from {head_players} to {len(player_paths)}')
    check(len(team_paths) == TEAM_COUNT, f'{len(team_paths)} team logs, expected {TEAM_COUNT}')

    check_logs('player logs', player_paths, changed, nba_source.GAMELOG_COLUMNS)
    check_logs('team logs', team_paths, changed)
    current_games = current_season_games(team_paths)
    check_league_tables(current_games)
    latest = check_meta()

    if args.smoke and not problems:
        smoke_test()

    if problems:
        print('Snapshot FAILED validation -- do not ship:')
        for problem in problems:
            print(f'  - {problem}')
        return 1
    print(f'Snapshot OK: {len(player_paths)} players, {current_games // 2} {SEASONS[-1]} games, '
          f'data through {latest}, {len(changed)} changed file(s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
