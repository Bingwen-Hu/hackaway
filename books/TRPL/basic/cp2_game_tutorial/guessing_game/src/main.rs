extern crate rand;

use std::io;
use std::cmp::Ordering; // enum
use rand::Rng; // trait

fn main(){
    println!("Guess the number!");
    
    // thread_rng: give us the particular random number generator that 
    // we're going to use: one that is local to the current thread of 
    // execution and seeded by the operation system.
    let secret_number = rand::thread_rng().gen_range(1, 101);

    // println!("The secret number is {}", secret_number);

    loop {
        println!("Please input your guess.");

        let mut guess = String::new();

        // io.stdin() return an instance `Stdin`
        // `&` means reference, `mut` make i
        // two line make it more readablet mutable
        // expect return a type io::Result
        // Result is an enum, and its variants are `Ok` and `Err`
        // If read_line success, expect return the value directly to 
        // the caller or an variable, else it cause program to crash
        io::stdin().read_line(&mut guess)
            .expect("Failed to read line"); 
        
        // shadow the same name for type convertion
        // turn a string to an integer u32
        let guess: u32 = match guess.trim().parse() {
            Ok(num) => num,
            Err(_) => continue,
        };

        // placeholder in order
        println!("Your guessed: {}", guess);

        match guess.cmp(&secret_number) {
            Ordering::Less => println!("Too small"),
            Ordering::Greater => println!("Too big!"),
            Ordering::Equal => {
                println!("You win!");
                break;
            }
        }
    }
}
