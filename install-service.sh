#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
UNIT=/etc/systemd/system/pixel-frame.service

case "$PROJECT_DIR" in
  *' '*|*'%'*) echo "Project path cannot contain spaces or %: $PROJECT_DIR" >&2; exit 1 ;;
esac

if [ "$(id -u)" -ne 0 ]; then
  echo "Run with sudo: sudo $0" >&2
  exit 1
fi

sed "s|@PROJECT_DIR@|$PROJECT_DIR|g" "$PROJECT_DIR/pixel-frame.service.in" > "$UNIT"
chmod 644 "$UNIT"
ln -sfn "$PROJECT_DIR/pixel-frame" /usr/local/bin/pixel-frame
systemctl daemon-reload
systemctl enable --now pixel-frame.service
echo "Installed and started pixel-frame.service"
