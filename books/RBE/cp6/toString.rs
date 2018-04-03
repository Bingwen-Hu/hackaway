// __str__ in Python

use std::string::ToString;

struct Circle {
    radius: i32,
}

impl ToString for Circle {
    fn to_string(&self) -> String {
        format!("Circle of radius {:?}", self.radius)
    }
}

/** demo for parse
 *  for every type, implement `FromStr` to make it parseable.
 */
fn parse_demo() {
    let parsed: i32 = "5".parse().unwrap();
    let turbo_parsed = "10".parse::<i32>().unwrap();
    let sum = parsed + turbo_parsed;
    println!("Sum: {:?}", sum);
}


fn main() {
    let circle = Circle { radius: 6 };
    println!("{}", circle.to_string());

    parse_demo();
}