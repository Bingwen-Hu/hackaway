// The Fn traits are provided by the standard library. 
// All closures implement at least one of the traits: Fn, FnMut, or FnOnce

// new example in TRPL
use std::thread;
use std::time::Duration;


// using closure instead of fn
// fn simulated_expensive_calculation(intensity: u32) -> u32 {
//     println!("calculating slowly...");
//     thread::sleep(Duration::from_secs(2));
//     intensity
// }

fn generate_workout(intensity: u32, random_number: u32) {
    let expensive_closure = |num| {
        println!("calculating slowly...");
        thread::sleep(Duration::from_secs(2));
        num
    };

    if intensity < 25 {
        // in the `if` branch, we call the cost function twice.
        println!(
            "TOday, do {} pushups", 
            expensive_closure(intensity)
        );
        println!(
            "TOday, do {} situps", 
            expensive_closure(intensity)
        );
    } else {
        if random_number == 3 {
            println!("Take a break today! Remember to stay hydrated!");
        } else {
            println!(
                "Today, run for {} minutes!",
                expensive_closure(intensity)
            );
        }
    }
}

fn main() {
    let simulated_user_specified_value = 10;
    let simulated_random_number = 7;

    generate_workout(
        simulated_user_specified_value,
        simulated_random_number
    );
}