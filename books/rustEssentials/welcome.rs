fn main () {
    println!("Welcome to the Game!");

    let mynum = abs(-4);
    println!("-4 abs is {}", mynum);
    on_linux();
    iloop();
}

fn abs(x: i32) -> i32 {
    if x > 0 {
        x
    } else {
        -x
    }
}

#[cfg(target_os = "linux")]
fn on_linux(){
    println!("This function works on linux");
}


#[test]
fn arithmetic(){
    if 2+3 == 5 {
        println!("you can calculate!");
    }
}


fn iloop(){
    'outer: loop {
        println!("Entered the outer loop! -");
        'inner: loop {
            println!("Entered the inner loop! -");
            // break;
            break 'outer;
        }
    }
    println!("Exited the loop!");
}
