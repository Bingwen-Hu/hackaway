solve the time syn problem in both-boot system (windows and linux)

ubuntu:
```
vim /etc/default/rcS

UTC=no
```

opensuse
```
sudo /usr/sbin/hwclock --systohc --localtime
```
