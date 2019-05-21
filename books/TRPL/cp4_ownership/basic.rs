// OwnerShip Rules
// 1. Each value in Rust has a variable that’s called its owner.
// 2. There can only be one owner at a time.
// 3. When the owner goes out of scope, the value will be dropped.

// Rust won’t let us annotate a type with the Copy trait if the type, or any of its parts, has implemented the Drop trait.
// For primitive, fixed size type, Copy is used.

fn main() {
    str_move();
    int_move();
}


fn int_move() {
    let x = 5;
    let y = x; // move?
    println!("value of x is {}", x);
    // In fact, there is no move, is copied
}

fn str_move() {
    fn calculate_length(s: String) -> usize {
        s.len()
    }
    let s1 = String::from("hello");
    let len = calculate_length(s1);
    println!("the length of stirng is {}", len);
    // can I access s1? The answer is No!
    // println!("The string is {}", s1); 
    // but using clone make a deep copy
    // what deep copy? not only reference but also data
    let s1 = String::from("hello");
    let len = calculate_length(s1.clone());
    println!("The string is {}", s1); 
}