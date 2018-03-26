/** Three types of struct
 * tuple struct, named tuples in Python
 * classic C structs
 * unit structs, useful to generics
 */

#[derive(Debug)]
struct Person<'a> {
    name: &'a str,
    age: u8,
}

// Unit Struct
struct Nil;

// Named Tuple
struct Pair(i32, f32);

// Used as a part of another struct
struct Point {
    x: f32,
    y: f32,
}

#[allow(dead_code)]
struct Rectangle {
    p1: Point,
    p2: Point,
}

fn main() {
    let name = "Peter";
    let age = 28;
    let peter = Person { name, age };
    
    println!("{:?}", peter);


    // instantiate a `Point`
    let point: Point = Point { x: 0.3, y: 0.4 };
    println!("point coordinates: ({}, {})", point.x, point.y);

    // struct update syntax
    let new_point = Point { x: 0.1, ..point};
    println!("second point: ({}, {})", new_point.x, new_point.y);

    // destructure the point using a `let` binding
    let Point { x: my_x, y: my_y } = point;

    let _rectangle = Rectangle {
        p1: Point { x: my_y, y: my_x },
        p2: point,
    };


    // instantiate a unit struct
    let _nil = Nil;

    // instantiate a tuple struct
    let pair = Pair(1, 0.1);

    // Access the fields of a tuple struct
    println!("pair contains {:?} and {:?}", pair.0, pair.1);

    // Destructure a tuple struct
    let Pair(integer, decimal) = pair;

    println!("pair contains {:?} and {:?}", integer, decimal);   
}