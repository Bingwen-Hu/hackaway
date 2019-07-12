pub fn roman_to_int(s: String) -> i32 {
    let mut ans = 0;
    let mut prev: char = ' '; 
    for c in s.chars() {
        match c {
            'I' => ans = ans + 1,
            'V' => {
                ans = ans + 5;
                if prev == 'I' { ans = ans - 2 * 1; }
            },
            'X' => {
                ans = ans + 10;
                if prev == 'I' { ans = ans - 2 * 1; }
            }
            'L' => {
                ans = ans + 50;
                if prev == 'X' { ans = ans - 2 * 10; }
            },
            'C' => {
                ans = ans + 100;
                if prev == 'X' { ans = ans - 2 * 10; }
            },
            'D' => {
                ans = ans + 500;
                if prev == 'C' { ans = ans - 2 * 100; }
            },
            'M' => {
                ans = ans + 1000;
                if prev == 'C' { ans = ans - 2 * 100; }
            },
            _ => {
                return ans; // invalid
            }
        }
        prev = c;
    }
    ans
}