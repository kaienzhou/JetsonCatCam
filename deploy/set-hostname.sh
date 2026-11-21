#!/usr/bin/env bash
# Give the board a short name so it answers to http://<name>.local:8080 on the local network.
# Run on the Jetson:   sudo bash deploy/set-hostname.sh [name]        (default: catcam)
set -e
NAME="${1:-catcam}"
case "$NAME" in
    ""|-*|*[!a-z0-9-]*) echo "name: lowercase letters, digits and - only (no spaces)"; exit 1 ;;
esac
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }

OLD="$(hostname)"
hostnamectl set-hostname "$NAME"
# sudo complains when its own host name does not resolve, so keep /etc/hosts in step
if grep -q '^127\.0\.1\.1' /etc/hosts; then
    sed -i "s/^127\.0\.1\.1.*/127.0.1.1\t$NAME/" /etc/hosts
else
    printf '127.0.1.1\t%s\n' "$NAME" >> /etc/hosts
fi
systemctl restart avahi-daemon 2>/dev/null || true
echo "host name: $OLD -> $NAME"
echo "on the local network: http://$NAME.local:8080"
