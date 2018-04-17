// demo combine Rc and RefCell
// Note:
// Rc allow multiple owner to the same immutable object
// RefCell allow mutate an object even there is immutable reference to it.

#[derive(Debug)]
enum List {
    // wrap to be a RefCell, so value in it can be interior mutated
    // RefCell wrapped to be a Rc, so as a whole multiple owner is allowed
    Cons(Rc<RefCell<i32>>, Rc<List>),
    Nil,
}

use List::{Cons, Nil};
use std::rc::Rc;
use std::cell::RefCell;

fn main() {
    // a value prepared to be modified
    let value = Rc::new(RefCell::new(5));

    // here, clone will increase the rc, no data really be copied.
    let a = Cons(value.clone(), Rc::new(Nil));
    let shared_list = Rc::new(a);

    let b = Cons(Rc::new(RefCell::new(6)), shared_list.clone());
    let c = Cons(Rc::new(RefCell::new(10)), shared_list.clone());

    *value.borrow_mut() += 10;

    println!("shared list after = {:?}", shared_list);
    println!("b after = {:?}", b);
    println!("c after = {:?}", c);
}