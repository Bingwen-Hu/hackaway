/** size of in C
 * 
 */

fn main() {
    let x = 1u8;
    let y = 'c'; // unicode....
    let z = 3;
    let f = 1.2_f32;

    println!("size of `x` in bytes: {}", std::mem::size_of_val(&x));
    println!("size of `y` in bytes: {}", std::mem::size_of_val(&y));
    println!("size of `z` in bytes: {}", std::mem::size_of_val(&z));
    println!("size of `f` in bytes: {}", std::mem::size_of_val(&f));
}