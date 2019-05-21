// We call having references as function parameters borrowing.
// mutable references have one big restriction: you can have only one mutable reference
// The benefit of having this restriction is that Rust can prevent data races at compile time. 
// Whew! We also cannot have a mutable reference while we have an immutable one

fn main() {
    annoy();
    prefer();
    prefer_mut();
    avoid_rare();
    imut_mut();
}

fn annoy() {
    let s1 = String::from("Happy");
    let (s2, len) = length(s1);
    println!("The length of '{}' is {}", s2, len);
    
    fn length(s: String) -> (String, usize) {
        let len = s.len();
        (s, len)
    }
}

fn prefer() {
    let s1 = String::from("Dream");
    let len = length(&s1);
    println!("The length of '{}' is {}", s1, len);

    fn length(s: &String) -> (usize) {
        s.len()
    }
}

fn prefer_mut() {
    let mut s = String::from("Mory");
    add_Jenny(&mut s);
    println!("{}", s);

    fn add_Jenny(s: &mut String) {
        s.push_str(" Loves Jenny");
    }
}


fn avoid_rare() {
    // using bracket to avoid borrowing at the same time.
    let mut s = String::from("Find the Good Word");
    {
        let borrow1 = &mut s;
        println!("borrow1 {}", borrow1)
    }
    let borrow2 = &mut s;
    println!("borrow2 {}", borrow2);
}

fn imut_mut() {
    let mut s = String::from("For your benefit");
    let s1 = &s;
    let s2 = &s;
    println!("make two immutable borrow!")
    // could not make this
    // let s3 = &mut s;
}