mod twosum;
mod longestsubstr;
mod reverseint;
mod atoi;
mod palindrone;
mod roman;
mod duplicate;

#[cfg(test)]
mod tests {
    #[test]
    fn test_two_sum() {
        let res = crate::twosum::two_sum([3, 2, 4].to_vec(), 6);
        assert_eq!([1, 2].to_vec(), res);
    }
    #[test]
    fn test_two_sum_hash() {
        let res = crate::twosum::two_sum_hash([3, 2, 4].to_vec(), 6);
        assert_eq!([1, 2].to_vec(), res);
    }

    #[test]
    fn test_longest_substr_a() {
        let res = crate::longestsubstr::length_of_longest_substring("abcdefd".to_string());
        assert_eq!(6, res);
    }
    #[test]
    fn test_longest_substr_b() {
        let res = crate::longestsubstr::length_of_longest_substring("abba".to_string());
        assert_eq!(2, res);
    }  
    #[test]
    fn test_reverse() {
        let res = crate::reverseint::reverse(123);
        assert_eq!(321, res);
        let res = crate::reverseint::reverse(2147483647);
        assert_eq!(0, res);
        let res = crate::reverseint::reverse(-123); 
        assert_eq!(-321, res);
        let res = crate::reverseint::reverse(-2147483648); 
        assert_eq!(0, res);
    }
    #[test]
    fn test_atoi() {
        let res = crate::atoi::atoi("123".to_string());
        assert_eq!(123, res);
        let res = crate::atoi::atoi("   123v".to_string());
        assert_eq!(123, res);
        let res = crate::atoi::atoi("fd123".to_string());
        assert_eq!(0, res);
        let res = crate::atoi::atoi("-123fdf".to_string());
        assert_eq!(-123, res);
        let res = crate::atoi::atoi("-2147483649fdf".to_string());
        assert_eq!(-2147483648, res);
        let res = crate::atoi::atoi("2147483649fdf".to_string());
        assert_eq!(2147483647, res);
        let res = crate::atoi::atoi("-91283472332".to_string());
        assert_eq!(-2147483648, res);
    }

    #[test]
    fn test_palindrone() {
        let res = crate::palindrone::is_palindrome(123);
        assert_eq!(false, res);
        let res = crate::palindrone::is_palindrome(121);
        assert_eq!(true, res);
        let res = crate::palindrone::is_palindrome(-121);
        assert_eq!(false, res);
    }

    #[test]
    fn test_roman_to_int() {
        let res = crate::roman::roman_to_int("III".to_string());
        assert_eq!(3, res);
        let res = crate::roman::roman_to_int("IVII".to_string());
        assert_eq!(6, res);
    }

    #[test]
    fn test_remove_duplicate() {
        let res = crate::duplicate::remove_duplicates(&mut [1, 1, 2, 2, 3, 3].to_vec());
        assert_eq!(3, res);
    }
}
