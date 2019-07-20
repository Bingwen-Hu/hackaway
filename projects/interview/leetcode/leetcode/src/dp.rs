// 我们开始学习动态规划吧


// https://leetcode-cn.com/problems/maximum-subarray
// 最大子序各，好像看不出什么动态规则的意味，反而像滑动窗口
pub fn max_sub_array(nums: Vec<i32>) -> i32 {
    let mut sum = nums[0];
    let mut ans = nums[0];

    for i in 1..nums.len() {
        if sum > 0 {
            // add positive sum means larger
            sum += nums[i];
        } else {
            // start from new one means larger
            sum = nums[i];
        }
        // ans always store the largest sum
        ans = std::cmp::max(sum, ans);
    }    
    ans
}


// https://leetcode-cn.com/problems/climbing-stairs/solution/
// basic dynamic programming
pub fn climb_stairs(n: i32) -> i32 {
    if n == 0 || n == 1 { 
        return 1;
    }

    // f(n) = f(n-1) + f(n-2)
    // iterative is harder than recursive
    let mut n_1 = 1; // f(n-1)
    let mut n_2 = 1; // f(n-2)
    let mut ans = 0;
    for _ in 1..n {
        ans = n_1 + n_2;
        n_1 = n_2;
        n_2 = ans; 
    }
    ans
}

// https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/solution/yi-ge-fang-fa-tuan-mie-6-dao-gu-piao-wen-ti-by-l-3/
// sell stock using state machine
// this is the solution for infinite k
pub fn max_profit_infinite(prices: Vec<i32>) -> i32 {
    let mut s_keep = std::i32::MIN; // you could not keep any stock on the very first day
    let mut s_empty = 0;

    for price in prices {
        s_keep = std::cmp::max(s_keep, s_empty - price); 
        s_empty = std::cmp::max(s_empty, s_keep + price);
    }
    return s_empty;
}

pub fn max_profit_once(prices: Vec<i32>) -> i32 {
    // suffix 0 means no trade (buy or sell) happen
    // 1 means it happend
    // let mut s_keep_0 = std::i32::MIN; // you could not keep any stock on the very first day
    let mut s_empty_0 = 0; 
    let mut s_keep_1 = std::i32::MIN; 
    let mut s_empty_1 = std::i32::MIN;

    for price in prices {
        s_keep_1 = std::cmp::max(s_keep_1, s_empty_0 - price); 
        s_empty_1 = std::cmp::max(s_empty_1, s_keep_1 + price);
    }
    return std::cmp::max(s_empty_1, 0);
}

pub fn max_profit_twice(prices: Vec<i32>) -> i32 {
    // suffix 0 means no trade (buy or sell) happen
    // 1 means it happend
    // let mut s_keep_0 = std::i32::MIN; // you could not keep any stock on the very first day
    let mut s_empty_0 = 0; 
    let mut s_keep_1 = std::i32::MIN; 
    let mut s_empty_1 = std::i32::MIN;
    let mut s_keep_2 = std::i32::MIN;
    let mut s_empty_2 = std::i32::MIN;

    for price in prices {
        s_keep_1 = std::cmp::max(s_keep_1, s_empty_0 - price); 
        s_empty_1 = std::cmp::max(s_empty_1, s_keep_1 + price);
        s_keep_2 = std::cmp::max(s_keep_2, s_empty_1 - price);
        s_empty_2 = std::cmp::max(s_empty_2, s_keep_2 + price);
    }
    return std::cmp::max(s_empty_2, 0);
}


// this one works but consume too much memory
pub fn max_profit_k_memory_consume(k: i32, prices: Vec<i32>) -> i32 {
    // from example above, we know the initial value is 0
    // here, k become a variable, some we need a matrix to
    // store different status
    // how many status we have?
    // empty or keep => 2
    // trade times => k
    // so we have 2k status
    let mut s_trade: [i32; 2] = [std::i32::MIN, std::i32::MIN]; // trade state: empty or keep
    let mut s_times: Vec<[i32;2]> = Vec::new(); 
    let k: usize = k as usize;
    for i in 0..k+1 {
        s_times.push(s_trade.clone());
    }
    s_times[0][0] = 0;
    for price in prices {
        for j in 0..k {
            s_times[j+1][1] = std::cmp::max(s_times[j+1][1], s_times[j][0] - price);
            s_times[j+1][0] = std::cmp::max(s_times[j+1][0], s_times[j+1][1] + price);
        }
    }
    return std::cmp::max(0, s_times[k][0]);        
}

// memory efficient version
pub fn max_profit_k(k: i32, prices: Vec<i32>) -> i32 {
    // here if k in unreasonable large, switch to infinite version
    if k > prices.len()/2 {
        return max_profit_infinite(prices);
    }

    let mut s_trade: [i32; 2] = [std::i32::MIN, std::i32::MIN]; // trade state: empty or keep
    let mut s_times: Vec<[i32;2]> = Vec::new(); 
    let k: usize = k as usize;
    for i in 0..k+1 {
        s_times.push(s_trade.clone());
    }
    s_times[0][0] = 0;
    for price in prices {
        for j in 0..k {
            s_times[j+1][1] = std::cmp::max(s_times[j+1][1], s_times[j][0] - price);
            s_times[j+1][0] = std::cmp::max(s_times[j+1][0], s_times[j+1][1] + price);
        }
    }
    return std::cmp::max(0, s_times[k][0]);        
}