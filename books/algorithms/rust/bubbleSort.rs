/**
 * 10, 34,  2, 45,  6,  8, 7,  4,  1,  3 
 *  1, 10, 34,  2, 45,  6, 8,  7,  4,  3
 *  1,  2, 10, 34,  3, 45, 6,  8,  7,  4
 * .......
 */

fn main() {
    let mut a = [10, 34, 2, 45, 6, 8, 7, 4, 1, 3];
    println!("{:10} {:?}", "Original:", a);
    

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

    println!("{:10} {:?}", "Sorted:", a)
}