
#[derive(Debug)]
struct Rectangle{
    length: u32,
    width: u32,
}


fn main() {

    basic();
    let width = 39;
    let length = 12;
    let rect1 = Rectangle{width: 59, length: 29};
    let rect2 = Rectangle{width, length};
    println!("rect1 is {:?}", rect1);
    println!("rect1 is {:#?}", rect2);
}

// Note 3
// 1. [derive(Debug)] is needed
// 2. when the var name and field name is same, a shortcut can be used.
// 3. {:#?} is a pprint.

fn basic() {
    #[derive(Debug)]
    struct User {
        username: String,
        email: String,
        sign_in_count: u64, 
        active: bool,
    }
    
    fn build_user(email: String, username: String) -> User {
        User {
            email, // shortcut
            username,
            active: true,
            sign_in_count: 1,
        }
    }

    // the whole variable should be mutable 
    let mut user1 = build_user("Mory@bodhicitta".to_string(), "sakyamory".to_string());
    let user2 = User {
        email: String::from("Memory@bodhicitta"),
        username: String::from("Jenny"),
        ..user1 // update shortcut
    };

    user1.email = String::from("Jenny2021@bodhicitta");
    println!("user1 info {:#?}", user1);
    println!("user1 info {:#?}", user2);
}