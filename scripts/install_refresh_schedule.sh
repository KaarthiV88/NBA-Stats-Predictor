#!/bin/bash
# Install, or reinstall, the daily snapshot refresh as a launchd agent.
#
#   scripts/install_refresh_schedule.sh              # every day at 10:00
#   scripts/install_refresh_schedule.sh 7 30         # every day at 07:30
#   scripts/install_refresh_schedule.sh --run-now    # start a run immediately
#   scripts/install_refresh_schedule.sh --uninstall
#
# If the Mac is asleep at the scheduled time, launchd runs the job on wake.
# Output goes to ~/Library/Logs/kv-money-market-refresh.log.

set -euo pipefail

LABEL="com.kvmoneymarket.snapshot-refresh"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/kv-money-market-refresh.log"
SCRIPT="$(cd "$(dirname "$0")" && pwd)/refresh_snapshot.sh"
DOMAIN="gui/$(id -u)"

case "${1:-}" in
  --uninstall)
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    rm -f "$PLIST"
    echo "Uninstalled $LABEL."
    exit 0
    ;;
  --run-now)
    launchctl kickstart "$DOMAIN/$LABEL"
    echo "Started. Follow along with: tail -f \"$LOG\""
    exit 0
    ;;
esac

HOUR="${1:-10}"
MINUTE="${2:-0}"

mkdir -p "$(dirname "$PLIST")" "$(dirname "$LOG")"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$SCRIPT</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>$HOUR</integer>
    <key>Minute</key>
    <integer>$MINUTE</integer>
  </dict>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>StandardOutPath</key>
  <string>$LOG</string>
  <key>StandardErrorPath</key>
  <string>$LOG</string>
</dict>
</plist>
EOF

launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
launchctl bootstrap "$DOMAIN" "$PLIST"
printf 'Installed %s: daily at %02d:%02d.\nLog: %s\n' "$LABEL" "$HOUR" "$MINUTE" "$LOG"
