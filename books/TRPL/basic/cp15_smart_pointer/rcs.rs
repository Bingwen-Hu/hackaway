// reference count

#[derive(Debug)]
enum List {
    Cons(i32, Rc<List>),
    Nil,
}

use List::{Cons, Nil};
use std::rc::Rc;

fn main() {
    let a = Rc::new(Cons(5, Rc::new(Cons(10, Rc::new(Nil)))));
    println!("rc = {}", Rc::strong_count(&a));
    let b = Cons(4, a.clone());
    println!("after add b to list, rc = {}", Rc::strong_count(&a));
    {
        let c = Cons(6, a.clone());
        println!("after add c to list, rc = {}", Rc::strong_count(&a));
    }
    println!("c out of scope, rc = {}", Rc::strong_count(&a));
}