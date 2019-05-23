// by default, variables in Rust is immutable

fn main(){
    let mut x = 5;
    println!("The value of x is {}", x);
    x = 19;
    println!("The value of x is {}", x);
    
    test_constant();
    shadow();
}

fn test_constant(){
    // declare in here is valid?
    // type of const must be announced
    // convention: All uppercase with underscore
    const MORY_CONSTANT: u32 = 100_000;

    println!("Mory has a constant {}", MORY_CONSTANT);
}

fn shadow(){
    // shadow allow you to use the same name and perform some modify before 
    // set some variable to inmmutable.
    // and shadow allow us to change type while `mut` could not.
    let x = 10;
    let x = x * 2;
    println!("In {}, value of x is {}", "THIS FILE", x);
}