fn main() {
    let x = 1;

    // copy trait here
    match x {
        1 | 2 => println!("one or two"),
        _ => println!("others"),
    }

    // copy trait here
    let x = 'b';
    match x {
        'A' ... 'Z' => println!("A through Z"),
        'a' ... 'j' => println!("a to j"),
        _ => println!("others..."),
    }
}