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

    // special format character. mimic %b in C
    println!("{} of b'{:b} people know binary, the other half doesn't", 1, 2);

    // mimic %3d in C
    for i in (1..4).rev() {
        print!("{number:3}", number=i);
    }
    println!("");
    // even more flexible, < means left alignment
    for i in (1..4).rev() {
        print!("{number:<width$}", number=i, width=i+1);
    }
    println!("");
    // padding with Zero, > means right alignment
    for i in (1..4).rev() {
        println!("{number:>0width$}", number=i, width=i+1);
    }

    // mimic %.3f in C
    let pi = 3.1415926;
    println!("Pi is {:0.3}", pi);
}