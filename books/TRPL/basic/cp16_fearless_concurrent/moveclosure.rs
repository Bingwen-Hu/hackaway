use std::thread;

fn main() {
    let v = vec![1, 2, 3];
    // move closure take ownership
    let handle = thread::spawn(move || {
        println!("{:?}", v);
    });

    handle.join();
}