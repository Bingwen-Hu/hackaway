use std::vec::Vec;
use std::collections::HashMap;

fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
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


fn two_sum_hash(nums: Vec<i32>, target: i32) -> Vec<i32> {
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

#[cfg(test)]
mod tests {
    #[test]
    fn test_two_sum() {
        let res = crate::two_sum([1, 2, 4, 6].to_vec(), 3);
        assert_eq!([0, 1].to_vec(), res);
        let res = crate::two_sum_hash([1, 2, 4, 6].to_vec(), 3);
        assert_eq!([0, 1].to_vec(), res);
    }
}

