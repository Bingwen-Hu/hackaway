use std::env;

fn main() {
    let args = env::args();
    println!("argv is {:?}", args);
    // collect is a function turn iterator into collections
    let args: Vec<String> = args.collect();
    println!("type of argv {:?}", args);
    let cuda = env::var("CUDA_VISIBLE_DEVICES").unwrap();
    println!("CUDA I use is {}", cuda);
}
