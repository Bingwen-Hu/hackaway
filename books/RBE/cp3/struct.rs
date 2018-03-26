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


fn rect_area(rect: &Rectangle) -> f32 {
    let &Rectangle { 
        p1: Point { x: x1, y: y1 },
        p2: Point { x: x2, y: y2 },
    } = rect;

    (x1 - x2) * (y1 - y2)
}

fn square(point: &Point, width: f32) -> Rectangle {
    let &Point { x: left, y: lower } = point;
    Rectangle { 
        p1: Point { x: left        , y: lower         },
        p2: Point { x: left + width, y: lower + width },
    }
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

    let rect = Rectangle {
        p1: Point { x: 3.0, y: 4.0 },
        p2: Point { x: 5.0, y: 7.0 },
    };

    let area = rect_area(&rect);
    println!("area of rectangle is {}", area);

    let point = Point { x: 4.0, y: 3.0 };
    let my_square = square(&point, 4f32);
    let area = rect_area(&my_square);
    println!("area of rectangle is {}", area);
}