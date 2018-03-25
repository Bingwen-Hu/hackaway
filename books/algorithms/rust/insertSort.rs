/** insert Sort in Rust
 * begin: 10  34  2  45  6  8  7  4  1  3 
 * sort:
 * 10 | 34  2  45  6  8  7  4  1  3 
 * 10  34 | 2  45  6  8  7  4  1  3 
 * 2  10 34 |  45  6  8  7  4  1  3 
 * .......
 */

fn main() {
    let mut a = [10, 34, 2, 45, 6, 8, 7, 4, 1, 3];
    for i in 1..a.len() {
        let mut j = i;
        let v = a[i];
        while (j > 0) && (a[j-1] > v) {
            a[j] = a[j-1];
            j = j - 1;
        }
        a[j] = v;
    }

    for i in a.iter() {
        print!("{:3}", i);
    }
    println!("");
}
