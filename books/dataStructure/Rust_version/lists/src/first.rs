/* define recursive data structure using Box */
pub struct List {
    head: Link,
}

enum Link {
    Empty,
    More(Box<Node>),
}

struct Node {
    elem: i32,
    next: Link,
}


/** 3 primary forms that self can take
 * ---------------------------------------------------------
 * Type      | meanings           | behaviours             |
 * ---------------------------------------------------------
 * self      | value              | will move              |
 * &mut self | mutable reference  | like reference in C++  |
 * &self     | shared reference   | like const reference   |
 * ---------------------------------------------------------
 * Note: special case can bypass &self
 */
impl List {
    pub fn new() -> Self {
        List { head: Link::Empty }
    }
}