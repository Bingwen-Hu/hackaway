/** classic consumer producer problem
 * 
 * 
 */

// assume wait(P) and signal(V) procedure
fn   wait(S: i32) {}
fn signal(S: i32) {}

fn main() {
    let mut mutex = 1;  // allow get in
    let mut empty = 10; // empty space 
    let mut full  = 0;  // nothing at the beginning
    let mut buffer: [i32; 10];
    let mut pindex = 0;
    let mut cindex = 0;
    // assume the following procedure execute paralelled
    
    // producer section:
    loop {
        let item = 123; // producer an item
        wait(empty);    // check for space
        wait(mutex);    // get the space, check for critical
        buffer[pindex] = item;
        pindex = (pindex + 1) % 10;
        signal(mutex);
        signal(full);
    }   

    loop {
        wait(full);
        wait(mutex);
        let item = buffer[cindex];
        cindex = (cindex + 1) % 10;
        signal(mutex);
        signal(empty);
    }

}