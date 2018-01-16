create table if not exists surpasser_power(
    area_id int(5) not null auto_increment,
    power_value int(5) default 0,
    area varchar(20) null,
    primary key (area_id)
);

insert into surpasser_power(power_value, area)
    values(3, 'China');
insert into surpasser_power(power_value, area)
    values(4, 'Europe');
insert into surpasser_power(power_value, area)
    values(5, 'Anndi');
insert into surpasser_power(power_value, area)
    values(3, 'America');
insert into surpasser_power(power_value, area)
    values(3, 'India');
