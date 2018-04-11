mod network {
    fn connect() {}
    // inside module
    mod client {
        fn connect() {}
    }
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
