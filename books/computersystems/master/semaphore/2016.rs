// N producer and M consumer share a buffer K. Rules as following:
// 1. only one process allowed to access buffer one time.
// 2. for every data in buffer, all the consumer has to get one.
// 3. when buffer full, producers blocked
// 4. when buffer emply, consumers blocked

// assume wait(P) and signal(V) procedure
fn   wait(S: i32) {}
fn signal(S: i32) {}


fn main() {
    let M = 10;

    let mut mutex = 1;  // allow to access buffer
    let mut empty = 10; // empty space 
    let mut full  = 0;  // nothing at the beginning
    let mut buffer: [i32; 10];
    let mut pindex = 0;
    let mut cindex = 0;

    let mut cc = M;
    let mut cc_mutex = 1;
    let mut cc_over = 1;
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

    // consumer
    loop {
        wait(full);                     // get something to eat!
        wait(mutex);                    // get the plate!
        wait(cc_mutex);                 // get RIGHT to access to cc
        let item = buffer[cindex];      // get food
        // If I am the first consumer come
        if cc == 10 {                   
            wait(cc_over);              
        }
        cc -= 1;                        // sign up
        // if i am the last consumer leave
        if cc == 0 {
            cindex = (cindex + 1) % 10;
            cc = 10;
            signal(cc_over);            
        }
        signal(cc_mutex);
        signal(mutex);
        signal(empty);
    }

}


// note: still broken