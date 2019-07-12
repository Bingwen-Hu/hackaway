pub fn is_palindrome(x: i32) -> bool {
    let mut x_mut = x;
    let mut ans = 0;

    if x < 0 { return false; }

    while x_mut != 0 {
        ans = ans * 10 + x_mut % 10;
        x_mut = x_mut / 10;
    }
    return x == ans;
}