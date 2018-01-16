# source mysql.sql
create table if not exists surpasser(
    surpasser_id int(20) not null auto_increment,
    name varchar(20) not null,
    surpass_kind varchar(20) default null,
    surpass_level varchar(20) default null,
    primary key (surpasser_id)
);


insert into surpasser(name, surpass_kind, surpass_level) 
    values("Mory", "Demon", "Highest");
insert into surpasser(name, surpass_kind, surpass_level) 
    values("Ann", "Light", "Highest");
insert into surpasser(name, surpass_kind, surpass_level) 
    values("Jenny", "Ice", "Medium");
insert into surpasser(name, surpass_kind, surpass_level) 
    values("Lydia", "Wind", "Highest");
insert into surpasser(name, surpass_kind, surpass_level) 
    values("Elena", "Soil", "High");
insert into surpasser(name, surpass_kind, surpass_level) 
    values("Ann Guevara", "Light", "Highest");
insert into surpasser(name, surpass_kind, surpass_level) 
    values("Yi dou", "Ice", "Highest");
