#!/usr/bin/expect

set timeout 5

set server ip
set user user
set remotepath /home/$user
set password passwd

set file [lindex $argv 0]

spawn scp -r $file $user@$server:$remotepath

expect "*:"

send "$password\r"

interact
