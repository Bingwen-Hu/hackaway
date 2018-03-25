/* binary search in Rust
 * 
 * 
 */
use std::io;


fn main() {
    let a = [1, 2, 3, 4, 6, 7, 8, 10, 34, 45];
 
    let mut input = String::new();
    io::stdin().read_line(&mut input)
        .expect("Fail to read.");
    
    let key: u32 = input.trim().parse()
        .expect("A integer is needed.");
    println!("key: {}", key);

    let mut begin = 0;
    let mut end = a.len();
    let mut found: bool = false;
    while end > begin {
        let mid = (begin + end) / 2;
        if a[mid] == key {
            println!("Found it! a[{}] = {}", mid, key);
            found = true;
            break;
        } else if a[mid] < key {
            begin = mid + 1;
        } else {
            end = mid - 1;
        }
    }
    if !found {
        println!("There is no {} in array.", key);
    }
}