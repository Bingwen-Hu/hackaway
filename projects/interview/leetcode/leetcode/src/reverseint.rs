// reverse integer
// 基本思路： 
// 将整数按个位拆来，一边拆一边组成新的数。每次生成下一个新数之前，都要先判断一下现在
// 的新数乘10之后会不会溢出。
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