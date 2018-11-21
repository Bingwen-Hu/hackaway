#!/usr/bin/expect

set timeout 5

spawn ssh $user@$ip -p $port

expect "*assword:"

send "$password\r"

interact

