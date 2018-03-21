fn create_string() {
    let mut s = String::from("Hello");
    s.push_str(", world!");
    println!("{}", s);
}

fn clone_string(){
    let s1 = String::from("hello");
    // should using clone or the ownership will move
    // and s1 is invalid
    let s2 = s1.clone();
    println!("s1 = {}, s2 = {}", s1, s2);
}

fn main(){
    create_string();
    clone_string();
}