/** 
 * advanced Option use
 * Generics
 * Lifetimes
 * Iterators
 */
use std::mem;

pub struct List {
    head: Link,
}

// yay type aliases!
// Link is a option type with Box of Node
// but Node has a type of Link! Recursive!
type Link = Option<Box<Node>>;

struct Node {
    elem: i32,
    next: Link,
}

impl List {
    pub fn new() -> Self {
        List { head: None }
    }

    pub fn push(&mut self, elem: i32) {
        let new_node = Box::new(Node {
            elem: elem,
            next: mem::replace(&mut self.head, Link::Empty),
        });
        self.head = Some(new_node);
    }

    pub fn pop(&mut self) -> Option<i32> {
        match mem::replace(&mut self.head, Link::Empty) {
            None => None,
            // move value from boxed_node first
            Some(node) => {
                let node = *node;
                self.head = node.next;
                Some(node.elem)
            }
        }
    }
}

// every type impl Drop will be deallocate after became junk.
impl Drop for List {
    fn drop(&mut self) {
        let mut cur_link = mem::replace(&mut self.head, None);
        while let Some(mut boxed_node) = cur_link {
            cur_link = mem::replace(&mut boxed_node.next, None);
        }
    }
}



#[cfg(test)]
mod test {
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
