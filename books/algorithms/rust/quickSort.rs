/** states quickSort in a clear way
 * 1. set a guard, such as the first element
 * 2. for every other element, if smaller, move to left, larger 
 * move to right
 * 3. swap the guard and the last index move.
 * 
 */

fn partition(a: &mut [u32; 12], p: usize, r: usize) {
    // Not the p is not the begin and r is not the length of A!
    // set the first element as a guard
    let x = a[p];
    let mut i = p+1;
    for j in p+1..r {
        if a[j] <= x {
            swap(a, i, j);
            i = i + 1;
        }
        println!("{:?}", a);
    }
    swap(a, p, i-1);
}


fn swap(a: &mut [u32; 12], i: usize, j: usize) {
    let t = a[i];
    a[i] = a[j];
    a[j] = t;
}

fn main() {
    println!("QuickSort in Rust!");
    let mut a = [11, 8, 12, 7, 9, 14, 5, 6, 16, 1, 3, 10];
    let len = a.len();
    println!("{:?}", a);
    partition(&mut a, 0, len);
    println!("{:?}", a);
}
