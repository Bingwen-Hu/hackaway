/** display traits implement for user-defined structure
 * 
 */

use std::fmt;

#[derive(Debug)]
struct MinMax(i64, i64);

// syntax: implement a trait
// like __str__ in Python
impl fmt::Display for MinMax {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

#[derive(Debug)]
struct Complex {
    real: f64, 
    imag: f64,
}

impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} + {}i", self.real, self.imag)
    }
}

fn main() {
    let minmax = MinMax(0, 42);

    println!("Compare strutures: ");
    println!("Display: {}", minmax);
    println!("Debug: {:?}", minmax);

    let complex = Complex { real: 4.4, imag: 2.2 };
    println!("Display: {}", complex);
    println!("Debug: {:?}", complex);
}