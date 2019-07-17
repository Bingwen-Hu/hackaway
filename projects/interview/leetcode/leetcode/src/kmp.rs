// kmp in Rust
pub fn str_str(haystack: String, needle: String) -> i32 {
    0        
}

pub fn compute_next(needle: String) -> Vec<usize> {
    let mut i: usize = 0;
    let mut j: usize = 0;
    let len = needle.len();
    let P: Vec<char> = needle.chars().collect();
    let mut next = vec![0];  

    while i < len-1 {
        if j == 0 || P[i] == P[j-1] {
            i += 1;
            j += 1;
            next.push(j);
        } else {
            j = next[j-1];
        }
    }
    println!("Finish compute next: \n");
    println!("{:?}", next);

    next
} 

