-- trigger is some function like callback function. (what's callback function?)

-- basic
create trigger newproduct after insert on products
for each row
begin
end;

-- delete
drop trigger newproduct;

-- create a table to store the log created by triggers
create table orders_log
(
   change_id    int      NOT NULL AUTO_INCREMENT,
   changed_on   datetime NOT NULL,
   changed_type char(1)  NOT NULL,
   order_num    int      NOT NULL,
   primary key (change_id)
) engine=maria;

-- Insert Trigger
create trigger neworder after insert on orders
for each row
begin 
   insert into orders_log(changed_on, change_type, order_num)
   values(now(), 'A', new.order_num);
end;

-- Delete Trigger NOTE that OLD refer to the row deleted!
create trigger deleteorder before delete on orders
for each row
begin
   insert into orders_log(changed_on, change_type, order_num)
   values(now(), 'D', OLD.order_num);
end;
