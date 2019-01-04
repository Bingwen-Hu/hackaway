# xset control the X server on Linux
xset led named "Scroll Lock"
xset -led named "Scroll Lock"

# xinput
```bash
# check device
xinput 

# enable certain device by device ID
xinput enable $ID

# disable certain device
xinput disable $ID
```

# xrandr
```bash
xrandr --newmode "1600x900_60.00"  118.25  1600 1696 1856 2112  900 903 908 934 -hsync +vsync
xrandr --addmode eDP-1 "1600x900_60.00"
xrandr -s 1600x900
```