#!/bin/bash
# Scheduled snapshot refresh: fetch the newest season, validate, commit only
# backend/data, push. Vercel redeploys from the push.
#
# Run by launchd (see install_refresh_schedule.sh); safe to run by hand too.
# Every failure path leaves the live site on its last good snapshot:
#   - nothing is pushed unless validate_snapshot.py passes;
#   - it never pushes code, only a data commit on top of origin/main;
#   - it skips (rather than guesses) when the checkout isn't in a state it can
#     safely commit from, and says why in a notification and the log.

set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND="$REPO/backend"
PYTHON="$BACKEND/venv/bin/python"
LOCK="$BACKEND/.refresh.lock"
STAMP="$BACKEND/.snapshot_refresh"
BUILD_ATTEMPTS=3

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
notify() {
  osascript -e "display notification \"$1\" with title \"KV Money Market data refresh\"" >/dev/null 2>&1 || true
}
finish() { log "$1"; notify "$1"; exit "${2:-0}"; }

# One run at a time. A lock older than 6 hours is from a crashed run or a reboot.
if [ -d "$LOCK" ] && [ -n "$(find "$LOCK" -maxdepth 0 -mmin +360)" ]; then
  rmdir "$LOCK"
fi
mkdir "$LOCK" 2>/dev/null || { log "Another refresh is running; exiting."; exit 0; }
trap 'rmdir "$LOCK"' EXIT

cd "$REPO" || finish "Repo not found at $REPO" 1
log "=== Refresh starting in $REPO ==="

# --- Preconditions: only ever add a data commit on top of origin/main -------
branch="$(git rev-parse --abbrev-ref HEAD)"
[ "$branch" = main ] || finish "Skipped: checkout is on '$branch', not main."
git fetch --quiet origin main || finish "Skipped: could not reach GitHub (git fetch failed)." 1
ahead="$(git rev-list --count origin/main..HEAD)"
[ "$ahead" = 0 ] || finish "Skipped: local main has $ahead unpushed commit(s); push or reset them first."
# The deployed backend must be the code that built the data.
dirty="$(git status --porcelain -- backend ':(exclude)backend/data')"
[ -z "$dirty" ] || finish "Skipped: uncommitted changes in backend/ -- commit or stash them first."
git merge --quiet --ff-only origin/main || finish "Skipped: could not fast-forward main to origin/main." 1

# --- Build (resumable, so retry after a cool-down if the API throttles) ------
built=0
for attempt in $(seq 1 "$BUILD_ATTEMPTS"); do
  log "Build attempt $attempt/$BUILD_ATTEMPTS"
  if (cd "$BACKEND" && caffeinate -i "$PYTHON" build_snapshot.py --refresh-current); then
    built=1
    break
  fi
  [ "$attempt" -lt "$BUILD_ATTEMPTS" ] && sleep 600
done
# The partial build stays on disk with its stamp, so the next run resumes it.
[ "$built" = 1 ] || finish "Build incomplete (API throttling?); nothing pushed. Next run resumes it." 1

# --- Validate; a bad build is discarded, never shipped ------------------------
if ! (cd "$BACKEND" && "$PYTHON" validate_snapshot.py --smoke); then
  git checkout --quiet -- backend/data
  git clean -fdq -- backend/data
  rm -f "$STAMP"
  finish "Refresh failed validation and was discarded; the site keeps its previous data." 1
fi

# --- Commit backend/data only, then push --------------------------------------
git add -A -- backend/data
if git diff --cached --quiet -- backend/data; then
  finish "No new data since the last refresh."
fi
through="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("latest_game_date"))' backend/data/meta.json)"
git commit --quiet -m "Refresh snapshot (games through $through)" -- backend/data \
  || finish "git commit failed; nothing pushed." 1
if ! git push --quiet origin main; then
  # Undo the local commit so the next run's "no unpushed commits" check passes;
  # the data stays in the working tree and is rebuilt next time.
  git reset --quiet HEAD~1
  finish "git push failed; nothing deployed. Next run retries." 1
fi
finish "Pushed new data (games through $through); Vercel is redeploying."
