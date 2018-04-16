fn basic() {
    let add_one = |x| x + 1;
    let five = add_one(4);
    assert_eq!(5, five);
}

fn basic2() {
    let calculate = |a, b| {
        let mut result = a * 2;
        result += b;
        result
    };

    assert_eq!(7, calculate(2, 3));
    assert_eq!(13, calculate(4, 5));
}

fn with_type() {
    let add_one = |x: i32| -> i32 { x + 1 };

    assert_eq!(2, add_one(1));
} 

// capture in other language
fn refer_env() {
    let x = 4;

    let equal_to_x = |z| z == x;

    let y = 4;

    assert!(equal_to_x(y));
}

fn main() {
    basic();
    basic2(); // with two parameters
    with_type();
    refer_env();
}