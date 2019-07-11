// reverse integer

pub fn reverse(x: i32) -> i32 {
    let mut ans = 0;
    let mut x = x;
    println!("x {}", x);
    while x != 0 {
        let pop = x % 10;
        if ans > std::i32::MAX / 10 || (ans == std::i32::MAX / 10 && pop > 7) {
            return 0;
        } 
        if ans < std::i32::MIN / 10 || (ans == std::i32::MIN / 10 && pop < -8) {
            return 0;
        }
        ans = ans * 10 + pop;
        x = x / 10;
        println!("ans {} pop {} x {}", ans, pop, x);
    }
    return ans;
}