extern crate rand;

use std::io;
use std::cmp::Ordering;
use rand::Rng;

fn main(){
    println!("Guess the number!");
    
    let secret_number = rand::thread_rng().gen_range(1, 101);

    println!("The secret number is {}", secret_number);

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
    
    let guess: u32 = guess.trim().parse()
        .expect("Please type a number!");

    // placeholder in order
    println!("Your guessed: {}", guess);

    match guess.cmp(&secret_number) {
        Ordering::Less => println!("Too small"),
        Ordering::Greater => println!("Too big!"),
        Ordering::Equal => println!("You win!"),
    }
    
}
