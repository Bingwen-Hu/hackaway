#[derive(Debug)]
struct Rectangle{
    width: u32,
    length: u32,
}


impl Rectangle{
    fn area(&self) -> u32 {
        self.length + self.width
    }

    fn exciting(&self) {
        println!("There is something exciting: \n{:#?}", self);
    }
}


fn main() {
    let rect1 = Rectangle{width: 84, length: 29};

    println!(
        "Let's compute the area of Rectangle: {}",
        rect1.area()
    );

    rect1.exciting();
}


// note that logic operator is && || just like C
