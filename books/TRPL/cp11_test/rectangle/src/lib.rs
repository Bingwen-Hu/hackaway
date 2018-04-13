#[derive(Debug)]
pub struct Rectangle {
    length: u32,
    width: u32,
}

// reference in Rust is like C++
impl Rectangle {
    pub fn can_hold(&self, other: &Rectangle) -> bool {
        self.length > other.length && self.width > other.width
    }
}           

pub fn add_two(a: i32) -> i32 {
    a + 2
}


pub fn greeting(name: &str) -> String {
    format!("Hello {}", name)
}

// should_panic guide
pub struct Guess {
    value: u32,
}

impl Guess {
    pub fn new(value: u32) -> Guess {
        if value < 1 {
            panic!("Guess value must be greater than or equal to 1, get {}", value);
        } else if value > 100 {
            panic!("Guess value must be less than or equal to 100, get {}", value);
        }
        Guess {
            value
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn larger_can_hold_smaller() {
        let larger = Rectangle { length: 8, width: 7 };
        let smaller = Rectangle { length: 5, width: 4 };

        assert!(larger.can_hold(&smaller));
    }

    #[test]
    fn smaller_cannot_hold_larger() {
        let larger = Rectangle { length: 8, width: 7 };
        let smaller = Rectangle { length: 4, width: 1 };

        assert!(!smaller.can_hold(&larger));
    }

    #[test]
    fn it_adds_two() {
        assert_eq!(4, add_two(2));
    }

    #[test]
    fn greeting_contains_name() {
        let result = greeting("carol");
        assert!(
            result.contains("carol"),
            "Greeting did not contains name, value was `{}`", "carol"
        );
    }

    #[test]
    #[should_panic]
    fn greater_than_100() {
        Guess::new(211);
    }
}
