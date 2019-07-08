// description:
// Given an array of integers, move all the zero behind 
// and keep other number in the same order

fn main() {
    let mut array: [i32; 13] = [19, 2, 0, 3, 43, 2, 0, 1, 12, 0, 22, 42, 12];
    test(&mut array);

    let mut array: [i32; 13] = [19, 2, 10, 3, 43, 2, 10, 1, 12, 10, 22, 42, 12];
    test(&mut array);
}

fn test(array: &mut [i32; 13]) {
    println!("Origin: {:?}", array);
    movezero(array);
    println!("Move: {:?}", array);

}

// very strange code
fn movezero(array: &mut [i32; 13]) {
    let len = 13;    
    let mut pos = 0;
    let mut count = 0;
    let mut start = 99;
    // this is a game to push all zero to the end
    // init pos, if there is no zero and all just end 
    for i in 0..len-1 {
        if array[i] != 0 {
            pos += 1;
        } else {
            count = 1;
            start = i + 1;
            break;
        } 
    }
    if start == 99 {
        return;
    }
    // let start push!
    for i in start..len {
        if array[i] != 0 {
            if pos+count >= len { break; }
            array[pos] = array[pos+count];
            array[pos+count] = 0;
            pos += 1;
        } else {
            count += 1;
        }
    }
}