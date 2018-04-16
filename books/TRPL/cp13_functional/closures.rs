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

fn main() {
    basic();
    basic2(); // with two parameters
}