// format print in Rust
// ----------------------------------------------------
// | format!   | write formatted text to `String`     |
// | print!    | same as format!, but send to console |
// | println!  | add newline on print!                |
// | eprint!   | print to stderr                      |
// | eprintln! | add newline on eprint!               |
// ----------------------------------------------------

fn main() {
    // Python style
    println!("{} days", 31);
    println!("{0}, this is {1}. {1}, this is {0}.", "Jenny", "Mory");
    println!("Name: {name}, best-friend: {bfriend}",
             name="Mory", bfriend="Ann");
}