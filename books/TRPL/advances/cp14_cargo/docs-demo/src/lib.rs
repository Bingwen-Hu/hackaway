//! # Docs Demo
//! 
//! `docs_demo` is a demo project to explain the usage of docs


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}


/// Adds one to the number given.
/// 
/// # Examples
/// 
/// ```
/// let arg = 5;
/// let answer = docs_demo::add_one(arg);
/// 
/// assert_eq!(6, answer);
pub fn add_one(x: i32) -> i32 {
    x + 1
}

// the usage of re-export makes it easier to user to use your crate
pub use self::kinds::PrimaryColor;

pub mod kinds {
    /// The primary colors according to the RGB color model
    pub enum PrimaryColor {
        Red, 
        Green,
        Blue,
    }
}