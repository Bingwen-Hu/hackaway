#[derive(Debug)]
struct CustomSmartPointer {
    data: String,
}


// drop will be called when object out of scope
impl Drop for CustomSmartPointer {
    fn drop(&mut self) {
        println!("Dropping customSmartPointer!");
    }
}

fn main() {
    let c = CustomSmartPointer { data: String::from("some data") };
    println!("CustomSmartPointer created.");
    println!("Wait for it...");
}