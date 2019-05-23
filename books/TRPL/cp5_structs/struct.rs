// Note
// 1. [derive(Debug)] is needed
// 2. when the var name and field name is same, a shortcut can be used.
// 3. {:#?} is a pprint.

// Unit-like structs can be useful in situations in which you need to 
// implement a trait on some type but don’t have any data that you want to store in the type itself. 

fn main() {
    basic();
    struct_tuple();
    unit_struct();
}


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

fn struct_tuple() {
    struct Color(u8, u8, u8);
    let c = Color(255, 10, 20);
    println!("Color is defined as struct tuple; c.1 is {}", c.1);
}

fn unit_struct() {
    #[derive(Debug)]
    struct UnitLike();
    let u = UnitLike();
    println!("I am a unit-like struct {:?}", u);
}