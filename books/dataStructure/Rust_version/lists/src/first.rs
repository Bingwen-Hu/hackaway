/* define recursive data structure using Box */
use std::mem;

// 0 overhead abstract
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

    pub fn push(&mut self, elem: i32) {
        let new_node = Box::new(Node {
            elem: elem,
            next: mem::replace(&mut self.head, Link::Empty),
        });
        // sad thing to note: we using mem::replace to set self.head Empty
        // before we set it a new value as the following line does.
        self.head = Link::More(new_node);
    }

    pub fn pop(&mut self) -> Option<i32> {
        match mem::replace(&mut self.head, Link::Empty) {
            Link::Empty => None,
            // move value from boxed_node first
            Link::More(boxed_node) => {
                let node = *boxed_node;
                self.head = node.next;
                Some(node.elem)
            }
        }
    }
}

// every type impl Drop will be deallocate after became junk.
impl Drop for List {
    fn drop(&mut self) {
        let mut cur_link = mem::replace(&mut self.head, Link::Empty);
        // while let means do this thing until this pattern fail
        // here, it means
        //     if cur_link is Link::More then
        //     boxed_node = *cur_link.More
        //     cur_link = boxed_node.next
        // loop it
        while let Link::More(mut boxed_node) = cur_link {
            cur_link = mem::replace(&mut boxed_node.next, Link::Empty);
        }
    }
}


/* add this means code of test only build when development and will not
build into final execute or library */
#[cfg(test)]
mod test {
    // because test is a new module, so List should be imported from 
    // super scope.
    use super::List;
    
    #[test]
    fn basic() {
        let mut list = List::new();
        
        // test for None
        assert_eq!(list.pop(), None);

        list.push(1);
        list.push(2);
        list.push(4);

        assert_eq!(list.pop(), Some(4));
        assert_eq!(list.pop(), Some(2));

        list.push(5);
        list.push(6);
        
        // test for None
        assert_eq!(list.pop(), Some(6));
        assert_eq!(list.pop(), Some(5));
        assert_eq!(list.pop(), Some(1));
        assert_eq!(list.pop(), None);

    }
}