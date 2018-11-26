echo 'PS1="\[$(ppwd)\]\[\033[1;34;1m\]\u@\h:\[\033[1;32;1m\]\w> \[\033[0m\]"' >> ~/.mory
echo 'export PIP="$HOME/.local/bin"' >> ~/.mory
echo 'export CARGO="$HOME/.cargo/bin"' >> ~/.mory
echo 'export CLJS="$HOME/softwares/cljs.jar"' >> ~/.mory
echo 'export PATH="$PIP:$CARGO:$PATH"' >> ~/.mory
echo 'alias python=python3' >> ~/.mory
echo 'alias deep="source $HOME/miniconda3/bin/activate deep"' >> ~/.mory