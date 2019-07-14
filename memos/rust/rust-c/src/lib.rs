extern {
    pub fn puts(s: *const u8) -> i32;
    fn hello();
    fn double_it(x: i32) -> i32;
    fn myatoi(x: *const u8) -> i32;
}


#[cfg(test)]
mod tests {
    #[test]
    fn test_atoi() {
        let string = b"123 fdf\0";
        unsafe { 
            let res = crate::myatoi(string.as_ptr()); 
            assert_eq!(123, res);
        }
    }
}