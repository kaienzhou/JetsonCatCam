#!/usr/bin/env bash
# Install JetsonCatCam as a systemd service: starts at boot, restarts when it exits.
# Run on the Jetson, from anywhere:   sudo bash deploy/install-service.sh
# Remove again with:                  sudo bash deploy/install-service.sh --remove
set -e
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_USER="${SUDO_USER:-$USER}"
UNIT=/etc/systemd/system/catcam.service

if [ "$1" = "--remove" ]; then
    systemctl disable --now catcam.service 2>/dev/null || true
    rm -f "$UNIT"
    systemctl daemon-reload
    echo "catcam service removed"
    exit 0
fi

[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
[ -f "$APP_DIR/config.json" ] || { echo "no config.json in $APP_DIR (copy config.example.json first)"; exit 1; }

sed -e "s#@USER@#$APP_USER#g" -e "s#@DIR@#$APP_DIR#g" "$APP_DIR/deploy/catcam.service.in" > "$UNIT"
systemctl daemon-reload
systemctl enable catcam.service
systemctl restart catcam.service
echo "installed for user $APP_USER in $APP_DIR"
echo "  status: systemctl status catcam      logs: journalctl -u catcam -f"
