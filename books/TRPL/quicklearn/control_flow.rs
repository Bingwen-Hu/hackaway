// condition and loops

fn if_else(number: i32){
    if number > 42 {
        println!("number {} larger than 42", number);
    } else if number < 7 {
        println!("number {} less than 7", number);
    } else {
        println!("number {} fit: 7 <= {} <= 42", number, number);
    }

    // if as a expression
    // shadow val `number`
    let number = if number > 21 { 42 } else { 7 };
    println!("My story you never know! I am {}", number);
}

fn while_loop(){
    let mut number = 4;
    while number != 0 {
        println!("{}!", number);
        number = number -1;
    }
    println!("Lift off!");
}

fn for_loop(){
    let a = [10, 20, 30, 44, 55];
    
    for element in a.iter() {
        println!("The value is: {}", element);
    }
    // arange
    for number in (1..10).rev() {
        println!("{}!", number);
    }
}

fn main(){
    println!("If else show: ");
    if_else(21);
    println!("\nwhile loop: ");
    while_loop();
    println!("\nfor loop: ");
    for_loop()
}