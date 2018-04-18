
fn main() {
    let closure = return_closure();
    println!("{}", closure(1));
}

fn return_closure() -> Box<Fn(i32) -> i32> {
    Box::new(|x| x + 1)
}