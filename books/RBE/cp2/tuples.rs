fn reverse(pair: (i32, bool)) -> (bool, i32) {
    let (integer, boolean) = pair;
    (boolean, integer)
}

#[derive(Debug)]
struct Matrix(f32, f32, f32, f32);

fn main() {
    let long_tuple = (1u8, 2u16, 3u32, 4u64,
                      -1i8, -2i16, -3i32, -4i64,
                      0.1f32, 0.2f64,
                      'a', true);

    // tuple index
    println!("long tupe first value: {}", long_tuple.0);
    println!("long tuple: {:?}", long_tuple);

    let matrix = Matrix(1.1, 1.2, 1.3, 1.4);
    println!("{:?}", matrix);

    // reverse
    let pair = (1, true);
    println!("revesed pair is: {:?}", reverse(pair));
}