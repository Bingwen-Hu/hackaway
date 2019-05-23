// match expression will move ownership

fn main() {
    let robot_name = Some(String::from("Bors"));

    match robot_name {
        Some(ref name) => println!("Found a name: {}", name),
        None => (),
    }

    println!("robot_name is: {:?}", robot_name);

    mutable_ref();
}


fn mutable_ref() {
    let mut robot_name = Some(String::from("Bors"));

    match robot_name {
        Some(ref mut name) => *name = String::from("Another name"),
        None => (),
    }
    println!("robot_name is: {:?}", robot_name);
}