// Another useful feature of impl blocks is that we’re allowed to define 
// functions within impl blocks that don’t take self as a parameter.

#[derive(Debug)]
struct Rectangle{
    width: u32,
    length: u32,
}


impl Rectangle{
    fn area(&self) -> u32 {
        self.length * self.width
    }

    fn exciting(&self) {
        println!("There is something exciting: \n{:#?}", self);
    }

    fn can_hold(&self, another: &Rectangle) -> bool {
        self.width > another.width && self.length > another.length
    }

    fn square(size: u32) -> Rectangle {
        Rectangle { width: size, length: size }
    }
}


fn main() {
    let rect1 = Rectangle{width: 84, length: 29};

    println!(
        "Let's compute the area of Rectangle: {}",
        rect1.area()
    );

    rect1.exciting();

    let rect2 = Rectangle{width: 32, length: 87};
    let hold = rect1.can_hold(&rect2);
    println!("Can Rect1 hold Rect2? {}", hold);

    let square = Rectangle::square(15);
    println!("Area of square is {}", square.area());
}
