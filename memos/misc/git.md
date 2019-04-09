### set alias
`git config --global alias.st status`

### auto store password and account
`git config credential.helper store`

### how to cancel last commit ?
`git rest HEAD~`

### trace large file
1. add package source

    `curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash `

2. install and init

   `sudo apt install lfs`

   `git lfs install`

3. trace certain extension file

    `git lfs track "model.weights"`

4. work as before 

### setup upstream
`git remote add upstream git@url`