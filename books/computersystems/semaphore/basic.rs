// classical syn problem
// Just for learning


let mut S = 0;

fn basic_P() {
    while S <= 0 {};
    S = S - 1;
}

fn basic_V() {
    S = S + 1;
}
//////////////////////////////////
//////////////////////////////////
struct Semaphore {
    value: i32, // init value means count of resources
    L: [i32; 10],
}

let mut S = Semaphore { value = 10, L = [-1; 50], };

fn  block(L: [i32; 100]) {}
fn wakeup(L: [i32; 100]) {}

fn record_P() {
    S.value = S.value - 1;
    if S.value < 0 {
        block(S.L);
    }
}

fn record_V() {
    S.value = S.value + 1;
    if S.value <= 0 {
        wakeup(S.L);
    }
}

fn main() {
    // this two mechanism is very weak
}