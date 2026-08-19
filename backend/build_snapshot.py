#!/usr/bin/env python3
"""Build the committed NBA data snapshot that the deployed backend reads.

Run this on a machine with a residential connection -- stats.nba.com silently
drops datacenter traffic, which is the whole reason the snapshot exists.

    python build_snapshot.py                 # build everything that's missing
    python build_snapshot.py --force         # rebuild from scratch
    python build_snapshot.py --players 25    # quick partial build, for testing
    python build_snapshot.py --workers 1     # slower, gentler on the API

The build is resumable: per-player files already on disk are skipped, so if the
API throttles you partway through, just run it again and it picks up where it
stopped.
"""

import argparse
import json
import logging
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# Force live mode regardless of the caller's environment -- building a snapshot
# by reading the snapshot would be a no-op, and a confusing one.
os.environ['NBA_SNAPSHOT'] = '0'
# A throttled request that blocks for two minutes stalls the entire build, so
# bulk fetching uses a much shorter timeout than the app does.
os.environ.setdefault('NBA_REQUEST_TIMEOUT', '20')

import nba_source  # noqa: E402  (must follow the env override above)

logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')
logging.getLogger('nba_source').setLevel(logging.ERROR)

SEASONS = ('2024-25', '2025-26')
SEASON_TYPES = ('Regular Season', 'Playoffs')

OUT = nba_source.SNAPSHOT_DIR

_print_lock = threading.Lock()
_calls = {'n': 0}

# When stats.nba.com rate-limits, every request starts timing out. Retrying
# through that just burns the clock, so back off globally: one worker's failure
# pauses all of them.
_throttle = {'streak': 0, 'until': 0.0}
_throttle_lock = threading.Lock()
THROTTLE_TRIP = 6        # consecutive failures before we call it throttling
THROTTLE_SLEEP = 90      # seconds to wait it out


def note_result(ok):
    """Record success/failure and trip a global pause when the API stops answering."""
    with _throttle_lock:
        if ok:
            _throttle['streak'] = 0
            return
        _throttle['streak'] += 1
        if _throttle['streak'] >= THROTTLE_TRIP and time.time() >= _throttle['until']:
            _throttle['until'] = time.time() + THROTTLE_SLEEP
            _throttle['streak'] = 0
            log(f'\n  ** API appears to be rate-limiting -- pausing {THROTTLE_SLEEP}s **\n')


def wait_if_throttled():
    while True:
        with _throttle_lock:
            remaining = _throttle['until'] - time.time()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 5))


def log(msg):
    with _print_lock:
        sys.stdout.write(msg + '\n')
        sys.stdout.flush()


def polite():
    """Randomised gap between calls so sustained fetching stays under the limit."""
    _calls['n'] += 1
    time.sleep(random.uniform(0.6, 1.2))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, separators=(',', ':'), default=str)
    tmp.replace(path)  # atomic, so an interrupted run never leaves half a file


def records(frame, columns=None):
    """DataFrame -> list of plain dicts, optionally trimmed to `columns`."""
    if frame is None or frame.empty:
        return []
    if columns:
        keep = [c for c in columns if c in frame.columns]
        frame = frame[keep]
    return json.loads(frame.to_json(orient='records', date_format='iso'))


def retrying(fn, *args, attempts=3, **kwargs):
    """Call fn with backoff, honouring the global throttle pause."""
    for attempt in range(1, attempts + 1):
        wait_if_throttled()
        try:
            result = fn(*args, **kwargs)
            note_result(True)
            return result
        except Exception as exc:  # noqa: BLE001 - any API failure is retryable here
            note_result(False)
            if attempt == attempts:
                log(f'    ! gave up: {type(exc).__name__}: {str(exc)[:70]}')
                return None
            time.sleep(3 * attempt)
    return None


# --- Per-player work -----------------------------------------------------

def build_player(player, force):
    """Write one player's game logs. Returns 'skipped' | 'ok' | 'empty' | 'failed'."""
    pid = player['id']
    path = nba_source.gamelog_path(pid)
    if path.exists() and not force:
        return 'skipped'

    blob = {}
    got_any = False
    for season in SEASONS:
        for season_type in SEASON_TYPES:
            frame = retrying(nba_source.player_game_log, pid, season, season_type)
            polite()
            if frame is None:
                return 'failed'
            rows = records(frame, nba_source.GAMELOG_COLUMNS)
            if rows:
                got_any = True
            blob[nba_source.season_key(season, season_type)] = rows

    write_json(path, blob)
    return 'ok' if got_any else 'empty'


def build_player_info(players_list, force):
    """One combined bio file for every player."""
    path = nba_source.player_info_path()
    existing = {}
    if path.exists() and not force:
        existing = json.loads(path.read_text(encoding='utf-8'))

    todo = [p for p in players_list if str(p['id']) not in existing]
    if not todo:
        log(f'  bios: already have all {len(existing)}')
        return

    log(f'  bios: fetching {len(todo)} ({len(existing)} cached)')
    done = 0
    for player in todo:
        frame = retrying(nba_source.player_common_info, player['id'])
        polite()
        if frame is not None and not frame.empty:
            existing[str(player['id'])] = json.loads(
                frame.iloc[[0]].to_json(orient='records', date_format='iso')
            )[0]
        done += 1
        if done % 50 == 0:
            write_json(path, existing)  # checkpoint so a crash isn't total loss
            log(f'    bios {done}/{len(todo)}')

    write_json(path, existing)
    log(f'  bios: {len(existing)} stored')


# --- League-wide + team tables ------------------------------------------

def build_league_tables(force):
    log('  league tables')
    for season in SEASONS:
        for season_type in SEASON_TYPES:
            path = nba_source.league_player_stats_path(season, season_type)
            if not path.exists() or force:
                frame = retrying(nba_source.league_player_stats, season, season_type)
                polite()
                write_json(path, records(frame))
                log(f'    player stats {season} {season_type}: {0 if frame is None else len(frame)} rows')

            # Opponent measure -- the only table carrying opponent points and
            # assists, which the league baselines need.
            opath = nba_source.league_opponent_stats_path(season, season_type)
            if not opath.exists() or force:
                frame = retrying(nba_source.league_opponent_stats, season, season_type)
                polite()
                write_json(opath, records(frame))
                log(f'    opponent stats {season} {season_type}: '
                    f'{0 if frame is None else len(frame)} rows')

            # Both windows the app asks for: default, and the 82-game window
            # predictive_model uses for opponent strength.
            for last_n in (None, '82'):
                tpath = nba_source.league_team_stats_path(season, season_type, last_n)
                if tpath.exists() and not force:
                    continue
                frame = retrying(nba_source.league_team_stats, season, season_type, last_n_games=last_n)
                polite()
                write_json(tpath, records(frame))
                log(f'    team stats {season} {season_type} last_n={last_n or "all"}: '
                    f'{0 if frame is None else len(frame)} rows')


def build_team_logs(force):
    from nba_api.stats.static import teams as static_teams
    all_teams = static_teams.get_teams()
    log(f'  team game logs ({len(all_teams)} teams)')
    for team in all_teams:
        path = nba_source.team_gamelog_path(team['id'])
        if path.exists() and not force:
            continue
        blob = {}
        for season in SEASONS:
            for season_type in SEASON_TYPES:
                frame = retrying(nba_source.team_game_log, team['id'], season, season_type)
                polite()
                blob[nba_source.season_key(season, season_type)] = records(frame)
        write_json(path, blob)
    log('  team game logs done')


# --- Entry point ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--force', action='store_true', help='rebuild files that already exist')
    parser.add_argument('--players', type=int, default=0, help='limit player count (testing)')
    parser.add_argument('--workers', type=int, default=2, help='parallel player fetches (default 2)')
    parser.add_argument('--ids', type=str, default='',
                        help='comma-separated player IDs to build (testing/repair)')
    args = parser.parse_args()

    from nba_api.stats.static import players as static_players
    roster = static_players.get_active_players()
    if args.ids:
        wanted = {int(x) for x in args.ids.split(',') if x.strip()}
        roster = [p for p in roster if p['id'] in wanted]
    elif args.players:
        roster = roster[:args.players]

    started = time.time()
    log(f'Building snapshot -> {OUT}')
    log(f'  {len(roster)} players, seasons {" + ".join(SEASONS)}, {args.workers} workers\n')

    build_league_tables(args.force)
    build_team_logs(args.force)

    log(f'  player game logs ({len(roster)})')
    tally = {'ok': 0, 'empty': 0, 'skipped': 0, 'failed': 0}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(build_player, p, args.force): p for p in roster}
        for i, future in enumerate(as_completed(futures), 1):
            player = futures[future]
            try:
                tally[future.result()] += 1
            except Exception as exc:  # noqa: BLE001
                tally['failed'] += 1
                log(f'    ! {player["full_name"]}: {exc}')
            if i % 25 == 0 or i == len(roster):
                log(f'    {i}/{len(roster)}  ok={tally["ok"]} empty={tally["empty"]} '
                    f'skip={tally["skipped"]} fail={tally["failed"]}')

    build_player_info(roster, args.force)

    write_json(OUT / 'meta.json', {
        'built_at': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        'seasons': list(SEASONS),
        'season_types': list(SEASON_TYPES),
        'player_count': len(roster),
        'api_calls': _calls['n'],
    })

    size = sum(f.stat().st_size for f in OUT.rglob('*.json'))
    mins = (time.time() - started) / 60
    log(f'\nDone in {mins:.1f} min · {size / 1024 / 1024:.1f} MB on disk · {_calls["n"]} API calls')
    if tally['failed']:
        log(f'{tally["failed"]} player(s) failed — re-run to retry just those.')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
