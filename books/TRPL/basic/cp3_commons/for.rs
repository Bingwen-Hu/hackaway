// have a look at for
fn main(){
    let array = [1, 2, 3, 4, 5];

    for i in array.iter() {
        println!("The element of array is: {}", i);
    }

    test_while();
    test_loop();
    test_iter();
}

fn test_while() {
    // no differece except the condition

    let mut i = 10;

    while i >= 0 {
        println!("I just learn the grammer: {}", i);
        i = i - 1;
    }
}

fn test_loop() {
    // return value from loop
    let mut counter = 0;
    let result = loop {
        counter += 1;
        if counter == 10 {
            break counter * 2;
        }
    };
    println!("Counter return from loop {}, {}", counter, result);
}

fn test_iter() {
    for number in (1..4).rev() {
        println!("{}!", number);
    }
}