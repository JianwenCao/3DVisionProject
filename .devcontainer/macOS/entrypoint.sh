#!/bin/bash
set -e

echo "Ensuring Xauthority exists..."
touch ~/.Xauthority

echo "Creating default xstartup script..."
mkdir -p ~/.vnc
cat << EOF > ~/.vnc/xstartup
#!/bin/sh

unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS

export XKL_XMODMAP_DISABLE=1
export XDG_CURRENT_DESKTOP="GNOME-Flashback:GNOME"
export XDG_MENU_PREFIX="gnome-flashback-"

[ -x /etc/vnc/xstartup ] && exec /etc/vnc/xstartup
[ -r \$HOME/.Xresources ] && xrdb \$HOME/.Xresources
xsetroot -solid "#2E3440"  # Set a modern background color
vncconfig -iconic &

# Start GNOME Flashback and UI components
dbus-launch --exit-with-session gnome-session --builtin --session=gnome-flashback-metacity --disable-acceleration-check --debug &
gnome-panel &
gnome-settings-daemon &
nautilus &
terminator -e "bash" &
EOF
chmod +x ~/.vnc/xstartup

echo "Starting VNC server on display :1..."
vncserver :1 -geometry 1280x800 -depth 24 -SecurityTypes None

echo "Starting noVNC on localhost:8080..."
/opt/novnc/utils/novnc_proxy --vnc localhost:5901 --listen localhost:8080

tail -f /dev/null
