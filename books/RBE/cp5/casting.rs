/** using `as` explicit convert type
 * 
 * 
 */
#[allow(overflowing_literals)]

fn main() {
    let decimal = 65.3214_f32;

    // explicit conversion
    let integer = decimal as u8;
    let character= integer as char;

    println!("Casting: {} -> {} -> {}", decimal, integer, character);

    // truncate highest bits
    println!("1000 as a u8 is {:>10b}", 1000);
    println!("1000 as a u8 is {:>10b}", 1000 as u8);

}