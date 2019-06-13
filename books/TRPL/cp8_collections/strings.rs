fn construct() {
    let s = String::new();

    let data = "initial contents";
    let s = data.to_string();

    let s = "initial contents".to_string();

    let s = String::from("initial contents");
}


fn update() {
    let mut s = String::from("Foo");
    let s2 = "bar";
    s.push_str(s2); // ownership is kept
    s.push('s');
    println!("{}", s);
    println!("{}", s2);
}

fn update_move() {
    let s1 = String::from("Hello ");
    let s2 = String::from("world");
    let s3 = s1 + &s2;  // s1 is moved
    println!("{}", s3);
    // println!("{}", s1); error
}

// format! is better because it is easier to read and 
// have not take any ownership of its parameters.
fn format() {
    let s1 = String::from("tic");
    let s2 = "Tac";
    let s3 = "toe".to_string();

    let s = format!("{}-{}-{}", s1, s2, s3);
    println!("{}\n{}-{}-{}", s, s1, s2, s3);
}

// important part
// indexing is not support
// string is wrapper of vector
fn strlen() {
    let len = String::from("Hola").len();
    println!("len of `hola` {}", len);
    let len = String::from("Здравствуйте").len();
    println!("len of `Здравствуйте` {}", len);
}

// in Rust, string bytes is stored and interpretted by program
fn deeper() {
    let s = "नमस्ते";
    for c in s.chars() {
        println!("{}", c);
    }

    for b in s.bytes() {
        println!("{}", b);
    }
}


fn main() {
    construct();
    update();
    update_move();
    format();
    strlen();
    deeper();
}