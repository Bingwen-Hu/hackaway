select *
from pop
where population <= 50;

select pop.country as country, gdp as gdp_in_millions_usd
from pop;

select * 
from gdp
where country in ("United States", "China");

select * 
from gdp 
where gdp between 10000 and 20000;
