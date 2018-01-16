select pop.country, pop.population, gdp.GDP
from pop. gdp
where pop.country = gdp.country;


select * 
from gdp 
where country in (select country from pop where population > 270);


