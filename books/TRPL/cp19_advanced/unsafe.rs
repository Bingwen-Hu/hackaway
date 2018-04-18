/** unsafe superpowers
1. Dereferencing a raw pointer
2. Calling an unsafe function or method
3. Accessing or modifying a mutable static variable
4. Implementing an unsafe trait
*/

fn main() {
    raw_pointer();
}

fn raw_pointer() {
    let mut num = 3;

    let r1 = &num as *const i32;        // immutable
    let r2 = &mut num as *mut i32;      // mutable

    // in unsafe block, immutable and mutable reference can 
    // exist at the same time.
    unsafe {
        println!("r1 is: {}", *r1);
        println!("r2 is: {}", *r2);
    }

    // call unsafe function in unsafe block
    unsafe {
        dangerous();
    }

}

unsafe fn dangerous() {}