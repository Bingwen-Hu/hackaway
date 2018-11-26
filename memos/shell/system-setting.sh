#!/usr/bin/expect

set timeout 10

# spawn sudo zypper refresh

# expect "*root:"
# send "2Foralfv\r"

# spawn sudo zypper update zypper
# spawn sudo zypper install git emacs sbcl rlwarp

# # vscode 
# spawn sudo rpm --import https://packages.microsoft.com/keys/microsoft.asc
# spawn sudo sh -c 'echo -e "[code]\nname=Visual Studio Code\nbaseurl=https://packages.microsoft.com/yumrepos/vscode\nenabled=1\ntype=rpm-md\ngpgcheck=1\ngpgkey=https://packages.microsoft.com/keys/microsoft.asc" > /etc/zypp/repos.d/vscode.repo'
# spawn sudo zypper refresh
# spawn sudo zypper install code

# rust
# spawn curl https://sh.rustup.rs -sSf | sh

# .emacs.d
# 

interact 