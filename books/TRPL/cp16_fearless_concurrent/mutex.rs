/* mutual exclusive */

use std::sync::Mutex;

fn main() {
    let m = Mutex::new(5);

    {
        // ask for m, lock() is blocked!
        let mut num = m.lock().unwrap();
        *num = 6;
    }

    println!("m = {:?}", m);
}