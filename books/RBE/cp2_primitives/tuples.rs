use std::fmt;

fn reverse(pair: (i32, bool)) -> (bool, i32) {
    let (integer, boolean) = pair;
    (boolean, integer)
}

fn transpose(matrix: Matrix) -> Matrix {
    let (x1, x2, y1, y2) = (matrix.0, matrix.1, matrix.2, matrix.3);
    Matrix(x1, y1, x2, y2)
}

#[derive(Debug)]
struct Matrix(f32, f32, f32, f32);

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "( {} {} )\n( {} {} )", self.0, self.1, self.2, self.3)
    }
}

fn main() {
    let long_tuple = (1u8, 2u16, 3u32, 4u64,
                      -1i8, -2i16, -3i32, -4i64,
                      0.1f32, 0.2f64,
                      'a', true);

    // tuple index
    println!("long tupe first value: {}", long_tuple.0);
    println!("long tuple: {:?}", long_tuple);

    let matrix = Matrix(1.1, 1.2, 1.3, 1.4);
    println!("{}", matrix);
    println!("{}", transpose(matrix));

    // reverse
    let pair = (1, true);
    println!("revesed pair is: {:?}", reverse(pair));
}