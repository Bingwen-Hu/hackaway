select rowid, * from table;

select distinct(name)
from namestable;


select count(*) as row_count
from gdp;


selct class, count(*) as countries_per_class
from classes
group by class;


select race, avg(tattoos, 'ct tattoos ever had')
from tattoos
group by race;


select distinct race, tattoos, 'year of birth' as birthyear, count(*) as weight
from tattoos
group by race, birthyear;


select *
from pop
order by country;


select * 
from pop
limit 20;


select * 
from pop 
limit 5 offset 3;

