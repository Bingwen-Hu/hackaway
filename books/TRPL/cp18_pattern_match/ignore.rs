fn main() {
    let numbers = (2, 4, 8, 16, 32);

    match numbers {
        (first, .., last) => {
            println!("Some numbers: {}, {}", first, last);
        },
    }

    struct_ignore();
}


#[derive(Debug)]
struct Point {
    x: i32,
    y: i32,
    z: i32,
}

fn struct_ignore() {
    let origin = Point { x: 0, y: 0, z: 0};
    match origin {
        Point { x, .. } => println!("x is {}", x),
    }
}