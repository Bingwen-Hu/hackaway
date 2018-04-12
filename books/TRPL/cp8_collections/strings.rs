fn construct() {
    let s = String::new();

    let data = "initial contents";
    let s = data.to_string();

    let s = "initial contents".to_string();

    let s = String::from("initial contents");
}


fn update() {
    let mut s = String::from("Foo");
    s.push_str(" bar");
    s.push('s');
    println!("{}", s);
}

fn update_move() {
    let s1 = String::from("Hello ");
    let s2 = String::from("world");
    let s3 = s1 + &s2;  // s1 is moved
    println!("{}", s3);
    // println!("{}", s1); error
}

fn format() {
    let s1 = String::from("tic");
    let s2 = "Tac";
    let s3 = "toe".to_string();

    let s = format!("{}-{}-{}", s1, s2, s3);
    println!("{}\n{}-{}-{}", s, s1, s2, s3);
}

fn main() {
    construct();
    update();
    update_move();
    format();
}