// 将字符串转成整数
pub fn atoi(string: String) -> i32 {
    println!("String: {}", string);
    let mut ans: i32 = 0;
    let mut start_parse: bool = false;
    let mut sign = 1; // 正负号
    for c in string.chars() {
        if c.is_whitespace() && start_parse == false { 
            continue; 
        } else if c == '-' && start_parse == false {
            start_parse = true; 
            sign = -1;
        } else if c == '+' && start_parse == false {
            start_parse = true;
        } else if let Some(i) = c.to_digit(10) {
            let i: i32 = i as i32;
            start_parse = true;
            if sign == 1 && ((ans > std::i32::MAX / 10) || (ans == std::i32::MAX / 10 && i > 7)) {
                return std::i32::MAX;
            } else if sign == -1 && ((ans * sign < std::i32::MIN / 10) || (sign * ans == std::i32::MIN / 10 && i > 8)) {
                return std::i32::MIN;
            } 
            ans = ans * 10 + i;
            println!("sign {} ans {} c {}", sign, ans, c);
        } else {
            break;
        }
    }
    ans * sign
}