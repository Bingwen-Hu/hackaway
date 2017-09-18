fn main() {
    let s1 = String::from("hello");

    let len = calculate_length(s1);

    println!("the length of stirng is {}", len);

    // can I access s1? The answer is No!
    // println!("The string is {}", s1); 
}


fn calculate_length(s: String) -> usize {
    s.len()
}
