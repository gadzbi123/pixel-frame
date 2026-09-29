#!/bin/sh
set -eu

if [ "$(id -u)" -ne 0 ]; then
  echo "Run with sudo from the account that should log in automatically: sudo $0" >&2
  exit 1
fi

LOGIN_USER=${SUDO_USER:-}
if [ -z "$LOGIN_USER" ] || [ "$LOGIN_USER" = root ]; then
  echo "Could not determine the non-root account. Run with sudo from that account." >&2
  exit 1
fi
case "$LOGIN_USER" in
  *[!a-zA-Z0-9_.-]*) echo "Unsupported account name: $LOGIN_USER" >&2; exit 1 ;;
esac
if ! getent passwd "$LOGIN_USER" >/dev/null; then
  echo "Unknown account: $LOGIN_USER" >&2
  exit 1
fi

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
LOGIN_UID=$(id -u "$LOGIN_USER")
case "$PROJECT_DIR" in
  *' '*|*'%'*) echo "Project path cannot contain spaces or %: $PROJECT_DIR" >&2; exit 1 ;;
esac

install -d -m 755 /etc/systemd/system/getty@tty1.service.d /etc/profile.d
cat > /etc/systemd/system/getty@tty1.service.d/override.conf <<EOF
[Service]
ExecStart=
ExecStart=-/sbin/agetty --autologin $LOGIN_USER --noclear %I linux
EOF
ln -sfn "$PROJECT_DIR/pixel-frame" /usr/local/bin/pixel-frame
cat > /etc/profile.d/pixel-frame-autostart.sh <<EOF
# Start Pixel Frame only for the configured account on the main Linux console.
case \$- in *i*) ;; *) return ;; esac
[ "\$(id -un)" = "$LOGIN_USER" ] || return
[ "\$(tty 2>/dev/null)" = /dev/tty1 ] || return
exec /usr/local/bin/pixel-frame run
EOF
SUDOERS_FILE=/etc/sudoers.d/pixel-frame
printf '%s ALL=(root) NOPASSWD: /usr/bin/python3 %s/pixel_frame.py --framebuffer\n' \
  "$LOGIN_USER" "$PROJECT_DIR" > "$SUDOERS_FILE"
chmod 440 "$SUDOERS_FILE"
visudo -cf "$SUDOERS_FILE"
chmod 644 /etc/systemd/system/getty@tty1.service.d/override.conf /etc/profile.d/pixel-frame-autostart.sh

# The foreground console session now owns the framebuffer, so don't also run
# a detached service for this account or the system service installed by
# install-service.sh.
USER_BUS=/run/user/$LOGIN_UID/bus
if [ -S "$USER_BUS" ] && runuser -u "$LOGIN_USER" -- env \
    XDG_RUNTIME_DIR="/run/user/$LOGIN_UID" \
    DBUS_SESSION_BUS_ADDRESS="unix:path=$USER_BUS" \
    systemctl --user cat pixel-frame.service >/dev/null 2>&1; then
  runuser -u "$LOGIN_USER" -- env \
    XDG_RUNTIME_DIR="/run/user/$LOGIN_UID" \
    DBUS_SESSION_BUS_ADDRESS="unix:path=$USER_BUS" \
    systemctl --user disable --now pixel-frame.service
fi
if systemctl cat pixel-frame.service >/dev/null 2>&1; then
  systemctl disable --now pixel-frame.service
fi
systemctl daemon-reload
systemctl restart getty@tty1.service
echo "Enabled tty1 auto-login for $LOGIN_USER and configured Pixel Frame to start there."
echo "Reboot to start the new login session on the main display."
