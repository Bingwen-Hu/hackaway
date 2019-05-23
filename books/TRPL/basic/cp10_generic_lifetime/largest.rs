// simple generic example in Rust

fn largest<T: PartialOrd + Copy>(list: &[T]) -> T {
    let mut largest = list[0];

    for &item in list.iter() {
        if item > largest {
            largest = item;
        }
    }
    largest
}

fn main() {
    let lst_f: [f64; 4] = [20.0, 1.3, 94f64, 3.4];
    let lst_i: [i32; 5] = [12, 32, 0, 34, 93];

    let largest_i = largest(&lst_i);
    let largest_f = largest(&lst_f);

    println!("largest int is {}", largest_i);
    println!("largest float is {}", largest_f);
}