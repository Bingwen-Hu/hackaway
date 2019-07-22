use std::collections::HashMap;

pub fn is_valid(s: String) -> bool {
    if s.len() == 0 { return true; }
    let mut map = HashMap::new();
    map.insert(')', '(');
    map.insert(']', '[');
    map.insert('}', '{');
    let mut stack: Vec<char> = Vec::new();
    for c in s.chars() {
        let pair = map.get(&c);
        match pair {
            Some(&p) => {
                if stack.len() == 0 {
                    return false;
                } else {
                    match stack.pop() {
                        Some(x) => {
                            if x != p { return false; }
                        },
                        _ => {},
                    }
                }
            },
            _ => stack.push(c),
        }
    }
    return stack.len() == 0;
}