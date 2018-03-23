// all types which want to use std::fmt formatting traits require
// an implementations to be printable.

// this structure cannot be printed either with `fmt::Display` 
// or with `fmt::Debug`
// struct UnPrintable(i32);

// like inherit from Debug
// #[derive(Debug)]
// struct DebugPrintable(i32);
#[derive(Debug)]
struct Person<'a> {
    name: &'a str,
    age: u8,
}

fn main() {
    let name = "Peter";
    let age = 26;
    let peter = Person { name, age };

    // Pretty print
    println!("{:#?}", peter);
}


// Print:
// ---------------
// debug  | {:?}
// pprint | {:#?}
// ---------------