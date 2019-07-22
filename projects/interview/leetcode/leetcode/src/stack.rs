use std::collections::HashMap;

pub fn is_valid(s: String) -> bool {
    if s.len() == 0 { return true; }
    let mut map = HashMap::new();
    map.insert(')', '(');
    map.insert(']', '[');
    map.insert('}', '{');
    let mut stack: Vec<char> = Vec::new();
    for c in s.chars() {
        match map.get(&c) {
            Some(&p) => match stack.pop() {
                Some(x) => { if x != p { return false; } },
                _ => return false,
            },
            _ => stack.push(c),
        }
    }
    return stack.len() == 0;
}