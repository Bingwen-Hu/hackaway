/** Note that:
 *  expressions do not include ending semicolons
 *  add a semicolon will omit returninig value
 */

fn five() -> i32 {
    5
}

fn main() {
    let x = five();
    println!("The value of x is: {}", x);
}