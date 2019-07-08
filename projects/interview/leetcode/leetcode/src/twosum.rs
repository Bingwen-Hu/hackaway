use std::vec::Vec;
use std::collections::HashMap;


pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
    for i in 0..nums.len()-1 {
        let res = target - nums[i];
        for j in i+1..nums.len() {
            if res == nums[j] {
                return vec![i as i32, j as i32];
            }
        }
    }
    return vec![];
}


pub fn two_sum_hash(nums: Vec<i32>, target: i32) -> Vec<i32> {
    let mut hash: HashMap<i32, i32> = HashMap::new();
    for (i, &v) in nums.iter().enumerate() {
        hash.insert(v, i as i32);
    }

    for (i_, &v) in nums.iter().enumerate() {
        if let Some(&i) = hash.get(&(target - v)){
            if i != i_ as i32 {
                return vec![i_ as i32, i as i32];
            }
        }
    }
    return Vec::new();
}
