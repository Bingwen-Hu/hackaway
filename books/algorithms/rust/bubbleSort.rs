/**
 * 10, 34,  2, 45,  6,  8, 7,  4,  1,  3 
 *  1, 10, 34,  2, 45,  6, 8,  7,  4,  3
 *  1,  2, 10, 34,  3, 45, 6,  8,  7,  4
 * .......
 */

fn main() {
    let mut a = [10, 34, 2, 45, 6, 8, 7, 4, 1, 3];
    print!("{:10}", "Original:");
    for i in a.iter() {
        print!("{:<3}", i);
    }
    println!("");

    let mut temp: i32; 
    for i in 1..a.len() {
        for j in (i..a.len()).rev() {
            if a[j] < a[j-1] {
                temp = a[j-1];
                a[j-1] = a[j];
                a[j] = temp;
            } 
        }
    }

    print!("{:10}", "Sorted:");
    for i in a.iter() {
        print!("{:<3}", i);
    }
    println!("");
}