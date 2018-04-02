/** semaphore apply: execute order
 *  S1 -> S2 -> S4 
 *    \    \     |
 *     S3   S5   |
 *      \    \   |
 *       \    \  |
 *        ----- S6
 */


fn wait (S: &mut i32) {
    while *S <= 0 {};
    *S = *S - 1;
}

fn signal(S: &mut i32) {
    *S = *S + 1;
}


fn main() {
    let mut a = 0;  // S2
    let mut b = 0;  // S3
    let mut c = 0;  // S4
    let mut d = 0;  // S5
    let mut e = 0;  // S6
    let mut f = 0;  // S6
    let mut g = 0;  // S6

    // assume the following procedure execute paralledly
    /* S1 */                 signal(&mut a); signal(&mut b);
    /* S2 */ wait(&mut a);   signal(&mut c); signal(&mut d);
    /* S3 */ wait(&mut b);   signal(&mut e);
    /* S4 */ wait(&mut c);   signal(&mut f);
    /* S5 */ wait(&mut d);   signal(&mut f);
}