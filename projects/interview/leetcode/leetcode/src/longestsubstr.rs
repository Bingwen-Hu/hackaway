// tips: we only need the length
// if we need the index, we need another two number 
// to store the index
use std::collections::HashMap;
use std::cmp;

pub fn length_of_longest_substring(s: String) -> i32 {
    let mut map: HashMap<char, u32> = HashMap::new();
    let mut start: u32 = 0;
    let mut ans: u32 = 0;

    for (end, c) in s.chars().enumerate() {
        if let Some(&i) = map.get(&c) {
            start = cmp::max(i, start);
        }
        ans = cmp::max(ans, (end as u32 - start + 1) as u32);
        map.insert(c, end as u32 + 1);
        println!("start {} end {} ans {} char {} map {:?}", 
            start, end, ans, c, map 
        );
    }        
    return ans as i32;
}


pub fn longest_common_prefix(strs: Vec<String>) -> String {
    let num = strs.len();
     

}