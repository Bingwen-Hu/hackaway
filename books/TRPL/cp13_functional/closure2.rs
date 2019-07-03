// The Fn traits are provided by the standard library. 
// All closures implement at least one of the traits: Fn, FnMut, or FnOnce

// new example in TRPL
use std::thread;
use std::time::Duration;
use std::collections::HashMap;


// using closure instead of fn
// fn simulated_expensive_calculation(intensity: u32) -> u32 {
//     println!("calculating slowly...");
//     thread::sleep(Duration::from_secs(2));
//     intensity
// }

struct Cacher<T>
    where T: Fn(u32) -> u32 
{
    calculation: T, // a lambda
    value_cache: HashMap<u32, u32>, // store the value of lambda execute
}

impl<T> Cacher<T>
    where T: Fn(u32) -> u32
{
    fn new(calculation: T) -> Cacher<T> {
        Cacher {
            calculation,
            value_cache: HashMap::new(),
        }
    }
    
    fn value(&mut self, arg: u32) -> u32 {
        match self.value_cache.get(&arg) {
            Some(&v) => v,
            None => {
                let v = (self.calculation)(arg);
                self.value_cache.insert(arg, v);
                v
            },
        }
    }
}

fn generate_workout(intensity: u32, random_number: u32) {
    let mut expensive_result = Cacher::new(|num| {
        println!("calculating slowly...");
        thread::sleep(Duration::from_secs(2));
        num
    });

    if intensity < 25 {
        // in the `if` branch, we call the cost function twice.
        println!(
            "TOday, do {} pushups", 
            expensive_result.value(intensity)
        );
        println!(
            "TOday, do {} situps", 
            expensive_result.value(intensity)
        );
    } else {
        if random_number == 3 {
            println!("Take a break today! Remember to stay hydrated!");
        } else {
            println!(
                "Today, run for {} minutes!",
                expensive_result.value(intensity)
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