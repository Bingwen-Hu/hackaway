/* mutual exclusive */
use std::thread;
// Arc means atomic rc, concurrent safely
use std::sync::{Mutex, Arc};


fn basic() {
    // mutex is a smart pointer
    let m = Mutex::new(5);

    {
        // ask for m, lock() is blocked!
        let mut num = m.lock().unwrap();
        *num = 6;
    }

    println!("m = {:?}", m);
}

fn shared() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();

            *num += 1;
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("result: {}", *counter.lock().unwrap());
}

fn main() {
    shared();
}