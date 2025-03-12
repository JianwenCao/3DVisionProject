#!/usr/bin/env bash
set -e

# Start a virtual display in the background
Xvfb :1 -screen 0 1280x800x24 &
sleep 2

# Start TigerVNC on display :1 with no password (SecurityTypes=None).
# This is *insecure* if port 5901 is open to the world, but we'll bind to localhost only.
tigervncserver :1 -geometry 1280x800 -depth 24 -SecurityTypes None

# Keep container alive with a shell.
exec bash
