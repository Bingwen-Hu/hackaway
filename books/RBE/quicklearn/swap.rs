
fn swap(a: &mut i32,b: &mut i32) {
    let t = *a;
    *a = *b; 
    *b = t;
}

fn main() {
    let mut a = 3;
    let mut b = 5;
    println!("a={} b={}", a, b);
    swap(&mut a, &mut b);
    println!("a={} b={}", a, b);
}