fn main() {
    let string1 = String::from("abcd");
    let string2 = "xyz";

    let result = longest(string1.as_str(), string2);
    println!("the longest string is {}", result);

    struct_with_lifetime();
}

// this could not compile because the lifttime of return value 
// is unknown
// define a generic lifetime annotations and solve the problem
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

fn struct_with_lifetime() {
    let novel = String::from("Call me Ishmael.");
    let first_sentence = novel.split('.')
        .next()
        .expect("Could not find a `.`");
    let i = ImportExcerpt { part: first_sentence };
    println!("{:?}", i);
}