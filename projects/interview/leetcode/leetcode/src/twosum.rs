use std::vec::Vec;
use std::collections::HashMap;


pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
    for (index, value) in nums.iter().enumerate() {
        let res = target - value;
        if res >= 0 {
            for (index_, value_) in nums[index..].iter().enumerate() {
                if res == *value_ {
                    return vec![index as i32, index_ as i32];
                }
            }
        }
    }
    return Vec::new();
}


pub fn two_sum_hash(nums: Vec<i32>, target: i32) -> Vec<i32> {
    let mut hash: HashMap<i32, i32> = HashMap::new();
    for (i, &v) in nums.iter().enumerate() {
        hash.insert(v, i as i32);
    }

    for (i_, &v) in nums.iter().enumerate() {
        if let Some(&i) = hash.get(&(target - v)){
            return vec![i_ as i32, i as i32];
        }
    }
    return Vec::new();
}
