// very simple example call function from C library
extern {
    pub fn puts(s: *const u8) -> i32;
    fn hello();
    fn double_it(x: i32) -> i32;
    fn myatoi(x: *const u8) -> i32;
}

fn main() {
    let x = b"hello world!\0";
    unsafe {
        puts(x.as_ptr());
        hello();
        let x = double_it(19);
        println!("{}", x);
        let x = b"42\0";
        let res = myatoi(x.as_ptr());
        println!("{}", res);
        let x = b"-42\0";
        let res = myatoi(x.as_ptr());
        println!("{}", res);
        let x = b"    -8147483649\0";
        let res = myatoi(x.as_ptr());
        println!("{}", res);
        let x = b"    2147483646\0";
        let res = myatoi(x.as_ptr());
        println!("{}", res);
        let x = b"   -2147483648\0";
        let res = myatoi(x.as_ptr());
        println!("{}", res);
    }
}