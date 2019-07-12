pub fn remove_duplicates(nums: &mut Vec<i32>) -> i32 {
    let len = nums.len();
    if nums.len() == 0 { return 0; }

    let mut k = 0; // 下标为0的数直接入选 
    for i in 1..len {
        let v = nums[i];
        if v != nums[k] {
            k += 1;
            nums[k] = v;
        }
    }
    (k + 1) as i32
}