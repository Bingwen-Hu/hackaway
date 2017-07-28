// by default, variables in Rust is immutable

fn main(){
    let mut x = 5;
    println!("The value of x is {}", x);

    x = 19;
    println!("The value of x is {}", x);
    

    test_constant();
}



fn test_constant(){
    // declare in here is valid?
    const MORY_CONSTANT: u32 = 100_000;

    println!("Mory has a constant {}", MORY_CONSTANT);
}


// Yes, const can be declared in any where.
