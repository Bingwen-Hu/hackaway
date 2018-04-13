// recoverable error

use std::fs::File;
use std::io::ErrorKind;

fn basic() {
    let f = File::open("panics.rs");

    let f = match f {
        Ok(file) => file,
        Err(error) => {
            panic!("There was a problem opening the file: {:?}", error);
        },
    };
}

fn multi_error() {
    let f = File::open("multi_error.rs");

    let f = match f {
        Ok(file) => file,
        Err(ref error) if error.kind() == ErrorKind::NotFound => {
            match File::create("multi_error.rs") {
                Ok(fc) => fc,
                Err(e) => {
                    panic!("Tried to create file but fail: {:?}", e);
                },
            }
        },
        Err(error) => {
            panic!("There was a problem opening the file {:?}", error);
        },
    };
}

fn main() {
    basic();
    multi_error();
}