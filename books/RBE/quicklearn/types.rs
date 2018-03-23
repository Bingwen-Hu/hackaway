// underscore use to make number readable

fn main(){
    println!("Rust types!");
    
    let guess: u32 = "42".parse().expect("Not a number!");
    let i: i32 = 12_345;
    println!("value of i is {}", i);
    println!("value of guess is {}", guess);
    println!("I am a binary number {}", 0b1111_0000);
    
    let b = true;
    println!("Rust has a boolean type? {}", b);
}