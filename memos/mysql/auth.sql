-- create a user  
create user mory@localhost identified by 'mory2016'

create database scrapy

Grant all on scrapy.* to 'mory'@'localhost'

-- change the password 
set password for root@localhost = password('linux')

-- log in
mysql -u mory -p scrapy
