fn basic() {
    panic!("crash and burn");
}

// a bug in our code leads to panic
fn backtraces() {
    let v = vec![1, 2, 3];
    v[100];
}

fn main() {
    // basic();
    backtraces();
}