/** classic consumer producer problem in C
 * 
 */

typedef int semaphore;

void   wait(semaphore s) {}
void signal(semaphore s) {}

void main() {
    semaphore mutex = 1;
    semaphore empty = 10;
    semaphore full  = 0;
    int buffer[10];
    int pindex = 0;
    int cindex = 0;

    // assume the following procedure execute parallell

    // producer section
    while (1) {
        int item = 1;       // produce one item!
        wait(empty);        // get space!
        wait(mutex);        // get access!
        buffer[pindex] = item;          // put into buffer! Done!
        pindex = (pindex + 1) % 10;     // update index! Done!
        signal(mutex);      // allow other access!
        signal(full);       // remaind some waiting consumer!
    }

    // I am a consumer, so I am waiting something to eat!
    while (1) {
        wait(full);         // food came out!
        wait(mutex);        // get the plate!
        int item = buffer[cindex];      // eat from index point to!
        cindex = (cindex + 1) % 10;     // next consumer!
        signal(mutex);      // allow other access!
        signal(empty);      // I have eat one! you can produce!
    }
}


// think: 
// is this a M consumer vs. N producer model?