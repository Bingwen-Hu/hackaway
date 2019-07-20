pub fn is_valid(s: String) -> bool {
    let mut stack: Vec<char> = Vec::new();
    let mut top = 0;
    
    for c in s.chars() {
        if top == 0 {
            stack.push(c);
        } else {
            match c {
                ')' => {
                    if stack[top] != '(' {
                        return false;
                    } else {
                        top -= 1;
                    }
                },
                '}' => {
                    if stack[top] != '{' {
                        return false;
                    } else {
                        top -= 1;
                    }
                },
                ']' => {
                    if stack[top] != '[' {
                        return false;
                    } else {
                        top -= 1;
                    }
                },
                _ => {
                    stack.push(c);
                    top += 1;
                }
            }  
        }
    }
    return top == 0;
}