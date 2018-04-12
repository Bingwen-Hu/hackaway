// vector can only store same type


fn construct() {
    let v: Vec<i32> = Vec::new();
    let mut v = vec![1, 2, 3];
}

fn vpush() {
    let mut v = Vec::new();
    v.push(5);
    v.push(1);
    v.push(6);
    print!("{:?}", v);
}

fn main() {
    construct();
    vpush();
}