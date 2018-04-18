fn main() {
    let v = vec![1, 2, 3];

    for (index, value) in v.iter().enumerate() {
        println!("{} is at index {}", value, index);
    }
}