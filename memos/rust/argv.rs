use std::env;

fn main() {
    let args = env::args();
    println!("argv is {:?}", args);
    // collect is a function turn iterator into collections
    let args: Vec<String> = args.collect();
    println!("type of argv {:?}", args);
}
