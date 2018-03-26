/** states quickSort in a clear way
 * 1. set a guard, such as the first element
 * 2. for every other element, if smaller, move to left, larger 
 * move to right
 * 3. swap the guard and the last index move.
 * 
 */

fn partition(mut a: &[u32; 12], p: u32, r: u32) {
    // Not the p is not the begin and r is not the length of A!
    // set the first element as a guard
    let x = a[p];

}


fn main() {
    println!("QuickSort in Rust!");
    let mut a = [2, 8, 12, 7, 9, 14, 5, 6, 16, 1, 3, 10];
    partition(&mut a, 0, a.len().count_ones());
}
