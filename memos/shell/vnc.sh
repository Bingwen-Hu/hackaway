# https://www.vandorp.biz/2012/01/installing-a-lightweight-lxdevnc-desktop-environment-on-your-ubuntudebian-vps/#.W_0qZ-EzZp9
# thanks for sharing

# Make sure Debian is the latest and greatest

apt-get update
apt-get upgrade
apt-get dist-upgrade

# Install X, LXDE, VPN programs

apt-get install xorg lxde-core tightvncserver

# Start VNC to create config file

tightvncserver :1

# Then stop VNC

tightvncserver -kill :1

# Edit config file to start session with LXDE:

nano ~/.vnc/xstartup

# Add this at the bottom of the file:
lxterminal &
/usr/bin/lxsession -s LXDE &

# Restart VNC

tightvncserver :1