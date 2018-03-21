/** 
 * advanced Option use
 * Generics
 * Lifetimes
 * Iterators
 */
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
            next: self.head.take(),
        });
        self.head = Some(new_node);
    }

    pub fn pop(&mut self) -> Option<i32> {
        self.head.take().map(|node| {
            let node = *node;
            self.head = node.next;
            node.elem
        })
    }
}

// every type impl Drop will be deallocate after became junk.
impl Drop for List {
    fn drop(&mut self) {
        let mut cur_link = self.head.take();
        while let Some(mut boxed_node) = cur_link {
            cur_link = boxed_node.next.take();
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
