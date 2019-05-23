#[derive(Debug)]
enum UsState {
    Alabama,
    Alaska,
    So,
    And,
    Soon,
    Etc,
    Haha,
}

enum Coin {
    Penny,
    Nickel,
    Dime,
    Quarter(UsState), // bind value here
}

// `_` works as a placeholder, which match any other thing
fn placeholder() {
    let some_value = 9u8;
    match some_value {
        1 => println!("One"),
        3 => println!("Three"),
        _ => println!("Not One or three"),
    }
}

fn value_in_cents(coin: Coin) -> i32 {
    match coin {
        Coin::Penny => {
            println!("Lucky Penny!");
            1
        },
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter(state) => {
            println!("State quarter from {:?}!", state);
            25
        },
    }
}


// if let, execute some code only when condition is met.
fn if_let() {
    let some_value = Some(3u8);
    if let Some(3) = some_value {
        println!("three");
    }
}

fn main() {
    let coin = Coin::Penny;
    let v = value_in_cents(coin);
    println!("{}", v);

    let coin  = Coin::Quarter(UsState::Haha);
    let v = value_in_cents(coin);
    println!("{}", v);

    placeholder();
    if_let();
}