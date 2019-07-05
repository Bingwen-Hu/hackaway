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
    // Rc::clone is preferred for its clear meaning
    // when use a.clone, we don't know whether it is a deep copy 
    // or just increase count of references.
    let b = Cons(4, Rc::clone(&a)); 
    println!("after add b to list, rc = {}", Rc::strong_count(&a));
    {
        let c = Cons(6, Rc::clone(&a));
        println!("after add c to list, rc = {}", Rc::strong_count(&a));
    }
    println!("c out of scope, rc = {}", Rc::strong_count(&a));
}