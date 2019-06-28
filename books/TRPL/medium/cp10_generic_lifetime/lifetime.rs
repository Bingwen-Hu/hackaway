// basic syntax
// &i32
// &'a i32
// &'a mut i32

// Ultimately, lifetime syntax is about connecting the lifetimes 
// of various parameters and return values of functions.

// Most of the time, the problem results from attempting to create 
// a dangling reference or a mismatch of the available lifetimes. 
// In such cases, the solution is fixing those problems, not 
// specifying the 'static lifetime.


// lifetime rule
// The first rule is that each parameter that is a reference gets its own lifetime parameter
// The second rule is if there is exactly one input lifetime parameter, 
//      that lifetime is assigned to all output lifetime parameters:
// The third rule is if there are multiple input lifetime parameters, but one of them is 
//      &self or &mut self because this is a method, the lifetime of self is assigned to 
//      all output lifetime parameters

fn main() {
    let string1 = String::from("abcd");
    let string2 = "xyz";

    let result = longest(string1.as_str(), string2);
    println!("the longest string is {}", result);

    struct_with_lifetime();
}

// angle bracket annotations
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

// struct hold reference needs a lifetime annotation
#[derive(Debug)]
struct ImportExcerpt<'a> {
    part: &'a str,
}

// rule 3 applied
impl<'a> ImportExcerpt<'a> {
    fn level(&self) -> i32 {
        3
    }
}

fn struct_with_lifetime() {
    let novel = String::from("Call me Ishmael.");
    let first_sentence = novel.split('.')
        .next()
        .expect("Could not find a `.`");
    let i = ImportExcerpt { part: first_sentence };
    println!("{:?}", i);
}