// slice is a view of the real data
// slice can be use as parameter
// slice .......


fn main() {
    let a = [1, 2, 3, 4, 5];
    
    let slice = &a[1..3];

    println!("the slice of a is {:?}", slice);
    

    let s = "haha";
    let len = length(s);
    
    println!("the length of s in {}", len);
    println!("Yes, {} is still accessible", s);
}


// slice as parameter

fn length(s: &str) -> usize {
    return s.len();
}
