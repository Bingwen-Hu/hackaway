fn main() {
    let mut s = String::from("hello");

    change(&mut s);

    println!("After change, s is {}", s);
}

fn change(s: &mut String){
    s.push_str("\tthe name is so stupid");
}
