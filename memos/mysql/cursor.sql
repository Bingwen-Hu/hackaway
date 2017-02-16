-- cursor can just be used in procedure.

-- ==================== basic
create procedure processorders()
begin 
   
   declare o int;   

   declare ordernumbers cursor
   for
   select order_num from orders;

   open ordernumbers;
   
   fetch ordernumbers into o;

   close ordernumbers;

end;
-- ====================


-- ==================== usage
create procedure processorders()
begin 
   
   declare done boolean default 0;
   declare o int;
   declare t decimal(8, 2);
   declare ordernumbers cursor
   for
   select order_num from orders;
   -- IMPORANT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   declare continue handler for sqlstate '02000' set done=1;

   open ordernumbers;
   
   create table if not exists ordertotals
          (order_num int, total decimal(8, 2));

   repeat
      fetch ordernumbers into o;
      call ordertotal(o, 1, t); --a procedure defined in procedure.sql
      insert into ordertotals(order_num, total) values(o, t);
      
   until done end repeat;

   close ordernumbers;

end;

-- ====================
