#!/bin/sh
# Show the frame on the main display at every boot. Run once: sudo ./install-service.sh
set -eu

if [ "$(id -u)" -ne 0 ]; then
  echo "Run with sudo: sudo $0" >&2
  exit 1
fi
OWNER=${SUDO_USER:-}
if [ -z "$OWNER" ] || [ "$OWNER" = root ]; then
  echo "Run with sudo from the account that owns the frame." >&2
  exit 1
fi
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
case "$PROJECT_DIR/$OWNER" in
  *[!a-zA-Z0-9_./-]*) echo "Unsupported characters in $PROJECT_DIR or $OWNER" >&2; exit 1 ;;
esac

sed -e "s|@PROJECT_DIR@|$PROJECT_DIR|g" -e "s|@USER@|$OWNER|g" \
  "$PROJECT_DIR/pixel-frame.service.in" > /etc/systemd/system/pixel-frame.service
sed "s|@USER@|$OWNER|g" "$PROJECT_DIR/pixel-frame.rules.in" > /etc/polkit-1/rules.d/50-pixel-frame.rules
chmod 644 /etc/systemd/system/pixel-frame.service /etc/polkit-1/rules.d/50-pixel-frame.rules
ln -sfn "$PROJECT_DIR/pixel-frame" /usr/local/bin/pixel-frame
# Remove the earlier login-shell autostart, which would start a second copy.
rm -f /etc/profile.d/pixel-frame-autostart.sh /etc/sudoers.d/pixel-frame

systemctl daemon-reload
systemctl enable pixel-frame.service
systemctl restart pixel-frame.service
echo "pixel-frame.service is running and starts at boot."
echo "Control it without sudo: pixel-frame stop | start | restart | status | logs"
