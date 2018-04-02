/** reader and writer problem
 * 1. when write, not allow read, when read not allow write
 * 2. many readers are allowed
 *
 */
fn   wait(S: i32) {}
fn signal(S: i32) {}

fn main() {
    let mut rmutex = 1;     // for readcount
    let mut wmutex = 1;     // for read and write procedure
    let mut readcount = 0;  // no reader at beginning
    // assume the follow procedure execute paralled
    
    // reader:
    loop {
        wait(rmutex);
        if readcount == 0 { 
            wait(wmutex);
            readcount = readcount + 1;
        }
        signal(rmutex);
        /** start reading 
         * after enjoy reading 
         * good time pass!
         */
        wait(rmutex);
        readcount = readcount - 1;
        if readcount == 9 {
            signal(wmutex);
        }
        signal(rmutex);
    }


    // writer
    loop {
        wait(wmutex);
        /** professional writer start
         *  write good articles .....
         */
        signal(wmutex);
    }
}