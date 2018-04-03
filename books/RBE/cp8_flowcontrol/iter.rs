/**
 * iter: just borrow ...
 * into_iter: move it!
 */

fn iter_borrow() {
    let names = vec!("Bob", "Frank", "Ferris");

    for name in names.into_iter() {
        match name {
            "Ferris" => println!("There is a rustacean among us!"),
            _ => println!("hello {}", name),
        }
    }
}

fn iter_move() {
    let names = vec!("Bob", "Frank", "Ferris");

    for name in names.into_iter() {
        match name {
            "Ferris" => println!("There is a rustacean among us!"),
            _ => println!("hello {}", name),
        }
    }
}

fn iter_mutable() {
    let mut names = vec!("Bob", "Frank", "Ferris");

    for name in names.iter_mut(){
        match name {
            &mut "Ferris" => println!("There is a rustacean among us!"),
            _ => println!("Hello {}", name),
        }
    }
}


fn main() {
    iter_borrow();
    iter_move();
    iter_mutable();
}