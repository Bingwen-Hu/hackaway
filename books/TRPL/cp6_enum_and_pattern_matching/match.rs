enum Coin {
    Penny,
    Nickel,
    Dime,
    Quarter,
}


fn value_in_cents(coin: Coin) -> i32 {
    match coin {
        Coin::Penny => {
            println!("Lucky Penny!");
            1
        },
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter => 25,
    }
}

fn main() {
    let coin = Coin::Penny;
    let v = value_in_cents(coin);
    println!("{}", v);

}