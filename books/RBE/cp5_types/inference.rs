/** smart inference */

fn main() {
    let elem = 5u8;

    // create an empty vector
    let mut vec = Vec::new();
    // at this time, type of Vec is unknown

    vec.push(elem);
    // now, type of vec is Vec<u8>

    println!("{:?}", vec);
}