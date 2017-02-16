-- how to call a procedure? (You still haven't created it!)

-- assume that we have a procedure call productpricing
call productpricing(@pricelow, @pricehigh, @priceaverage);


-- create a procedure
-- ====================
delimiter //

create procedure productpricing()
begin 
      select Avg(prod_price) as priceaverage
      from products;
end //

delimiter ;
-- ====================

-- delete a procedure
drop procedure productpricing;
drop procedure if exists productpricing;

-- using variable
-- ====================
create procedure productpricing(
   out pl decimal(8, 2);
   out ph decimal(8, 2);
   out pa decimal(8, 2);
)
begin 
   select Min(prod_price)
   into pl
   from products;
   select Max(prod_price)
   into ph
   from products;
   select Avg(prod_price)
   into pa
   from products;
end;
-- ====================

-- another example
-- ====================
create procedure ordertotal(
   in  onumber Int,
   out ototal decimal(8, 2)
)
begin 
   select Sum(item_price*quantity)
   from orderitems
   where order_num = onumber
   into ototal;
end;
-- ====================

call ordertotal(20005, @total);



-- another intelligence example
-- ====================
-- Name: ordertotal
-- Parameters: onumber = order number
--             taxable = 0 if not taxable, 1 if taxable
--             ototal  = order total variable
create procedure ordertotal(
   in  onumber Int,
   in  taxable Boolean,
   out ototal  Decimal(8, 2)
) -- obtaiin order total, optionally adding tax
begin
   -- Declare variable for total
   declare total decimal(8, 2)   
   -- Declare tax percentage
   declare taxrate Int default 6;

   -- Get the order total
   select sum(item_price*quantity)
   from orderitem
   where order_num = onumber
   into total;

   -- is this taxable?
   if taxable then
      select total + (total/100*taxrate) into total; --select into is the basic opt to change a value
   end if;

   select total into ototal;

end
-- ====================

-- check the procedure
show create procedure ordertotal;

show procedure status like 'ordertotal';
