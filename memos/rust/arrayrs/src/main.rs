#[macro_use(array, stack)]
extern crate ndarray;
use ndarray::{arr2, Axis};

fn basic() {
    let a1 = array![1, 2, 3, 4];
    let a2 = array![[1, 2],
                    [3, 4]];
    assert_eq!(a1.shape(), &[4]);
    assert_eq!(a2.shape(), &[2, 2]);
}

// so why IRust is important?
// easier to learn!
fn stack_d() {
    let a = arr2(&[[2, 2],
                   [3, 3]]);
    assert!(
        stack![Axis(0), a, a] 
            == arr2(&[[2, 2],
                      [3, 3],
                      [2, 2],
                      [3, 3]])
    );
}

fn main() {
    basic();
    stack_d();
}
