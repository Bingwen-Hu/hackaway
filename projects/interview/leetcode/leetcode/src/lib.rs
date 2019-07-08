mod twosum;

#[cfg(test)]
mod tests {
    #[test]
    fn test_two_sum() {
        let res = crate::twosum::two_sum([1, 2, 4, 6].to_vec(), 3);
        assert_eq!([0, 1].to_vec(), res);
        let res = crate::twosum::two_sum_hash([1, 2, 4, 6].to_vec(), 3);
        assert_eq!([0, 1].to_vec(), res);
    }
}

