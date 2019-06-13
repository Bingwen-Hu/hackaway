// Another data type that does not have ownership is the slice
// A string slice is a reference to part of a String
// For the purposes of introducing string slices, we are assuming ASCII only in this section
fn main() {
    str_slice();    
    int_slice();

    let string = String::from("Mory and Ann");
    let word = first_word(&string); // deref coercion
    println!("First word of string '{}' is {}", string, word);
}


// slice as parameter
fn str_slice() {
    let s = "haha"; // type of s is &str
    let len = length(s);
    
    println!("the length of s in {}", len);
    println!("Yes, {} is still accessible", s);

    fn length(s: &str) -> usize {
        return s.len();
    }
}


fn int_slice() {
    let a = [1, 2, 3, 4, 5];
    let slice = &a[1..3];
    println!("the slice of int array is {:?}", slice);
}


// this signature accept string slice and String type
fn first_word(s: &str) -> &str {
    let bytes = s.as_bytes();
    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return &s[0..i];
        }
    }
    &s[..]
}