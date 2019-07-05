// refcell allow interior mutability
// interior mutability means mutating data even though there are 
// immutable references to that data.

// the RefCell<T> type represents single ownership over the data it holds. 
// So, what makes RefCell<T> different from a type like Box<T>?
// With references and Box<T>, the borrowing rules’ invariants are enforced 
// at compile time. With RefCell<T>, these invariants are enforced at runtime. 
// With references, if you break these rules, you’ll get a compiler error. 
// With RefCell<T>, if you break these rules, your program will panic and exit.

// something strange will happen!

use std::cell::RefCell;

fn a_fn_that_immutably_borrows(a: &i32) {
    println!("a is {}", a);
}

fn a_fn_that_mutably_borrows(b: &mut i32) {
    *b += 1;
}

fn demo(r: &RefCell<i32>) {
    a_fn_that_immutably_borrows(&r.borrow());
    a_fn_that_mutably_borrows(&mut r.borrow_mut());
    a_fn_that_immutably_borrows(&r.borrow());
}

fn main() {
    // even data is immutable
    let data = RefCell::new(5);
    // value is change
    demo(&data);
}