// test case for enum
use List::*;

enum List {
    Cons(u32, Box<List>),
    Nil,
}

impl List {
    // return an empty list
    fn new() -> List {
        Nil
    }
    // append a new elem to the list
    fn prepend(self, elem: u32) -> List {
        Cons(elem, Box::new(self))
    }

    // recursive count the length
    fn len(&self) -> u32 {
        match *self {
            Cons(_, ref tail) => 1 + tail.len(),
            Nil => 0
        }
    }

    // recursive print
    fn stringify(&self) -> String {
        match *self {
            Cons(head, ref tail) => {
                format!("{}, {}", head, tail.stringify())
            },
            Nil => format!("Nil")
        }
    }
}


fn main() {
    let mut list = List::new();

    list = list.prepend(1);
    list = list.prepend(2);
    list = list.prepend(4);

    println!("Linked list has length: {}", list.len());
    println!("{}", list.stringify());
}