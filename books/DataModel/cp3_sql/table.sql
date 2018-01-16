begin:
    create table newtab(name, age);
    insert into newtab values("Joe", 12);
    insert into newtab values("Jill", 14);
    insert into newtab values("Bob", 14);
commit;


create table tourist_traps as
    select country
    from lonely_planet
    where (0.0+pp) > 600;


drop table newtab;


create table gdp2 as select * from gdp;
delete from gdp where gdp=12345;

insert into some_table select * from another_table;
update pop set population=23456 where country="Irap";


