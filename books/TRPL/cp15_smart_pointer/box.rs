// situations use box
//
// 1. When you have a type whose size can’t be known at compile time and you 
//    want to use a value of that type in a context that requires an exact size
// 2. When you have a large amount of data and you want to transfer ownership 
//    but ensure the data won’t be copied when you do so
// 3. When you want to own a value and you care only that it’s a type that 
//    implements a particular trait rather than being of a specific type



// The Box<T> type is a smart pointer because it implements the Deref trait, 
// which allows Box<T> values to be treated like references
fn main() {
    let b = Box::new(5);
    println!("b = {}", b);
}