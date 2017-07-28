// how functions works?

fn main() {
    let s = fn_with_return();

    println!("the return value is {}", s);
    
}


fn fn_with_return() -> String {
    let ret = String::from("Mory hope he can succeed in both 
in-life and out-life");

    // without a semicolon, the value is return
    ret
}


// note that: In Rust, string is naturally support new line
