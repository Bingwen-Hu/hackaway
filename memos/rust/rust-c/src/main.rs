// very simple example call function from C library
extern {
    pub fn puts(s: *const u8) -> i32;
    fn hello();
}

fn main() {
    let x = b"hello world!\0";
    unsafe {
        puts(x.as_ptr());
        hello();
    }
}