// less support in Rust

use std::collections::HashMap;


fn construct() {
    let mut scores = HashMap::new();
    scores.insert(String::from("Blue"), 10);
    scores.insert(String::from("Yellow"), 20);    
    println!("{:?}", scores);

    let teams = vec![String::from("Blue"), String::from("Yellow")];
    let scores = vec![10, 50];

    let scores: HashMap<_, _> = teams.iter().zip(scores.iter()).collect();
    println!("{:?}", scores);
}


fn mget() {
    let mut scores = HashMap::new();
    scores.insert(String::from("Blue"), 20);
    scores.insert(String::from("Yellow"), 50);


    let team_name = String::from("Blue");
    let score = scores.get(&team_name);         // no move
    match score {
        Some(v) => println!("score of {} is {}", team_name, v),
        _ => (),
    }

    // nice syntax
    for (key, value) in &scores {
        println!("{}: {}", key, value);
    }
}

fn update() {
    let mut scores = HashMap::new();
    scores.insert(String::from("Blue"), 20);
    scores.insert(String::from("Yellow"), 50);
    // insert twice will update the value
    scores.insert(String::from("Blue"), 10);
    println!("{:?}", scores);

    // insert only no value 
    scores.entry(String::from("Blue")).or_insert(20);
    scores.entry(String::from("Red")).or_insert(20);
    println!("{:?}", scores);

    // update base on old value
    {
        let score = scores.entry(String::from("Blue")).or_insert(0);
        *score += 1;
    } // mutable borrow score go out of scope
    println!("{:?}", scores);
}


fn main() {
    construct();
    mget();
    update();
}