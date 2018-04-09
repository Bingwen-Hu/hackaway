
#[derive(Debug)]
struct Rectangle{
    length: u32,
    width: u32,
}


fn main() {

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
