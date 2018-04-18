/* a trait that requires Self to be Sized is not allowed to be a trait object.
*/

// define a public trait
pub trait Draw {
    fn draw(&self);
}



pub struct Screen {
    // components holds a vector of trait object that implement the Draw trait.
    pub components: Vec<Box<Draw>>,
}


impl Screen {
    pub fn run(&self) {
        for component in self.components.iter() {
            component.draw();
        }
    }
}

/* here, both Button and SelectBox implement draw 
   this is duck type in Rust!
   duck type needs dynamic dispatch and have a 
   runtime cost
*/

pub struct Button {
    pub width: u32,
    pub height: u32,
    pub label: String,
}

impl Draw for Button {
    fn draw(&self) {
        println!("Draw a button");
    }
}


struct SelectBox {
    width: u32,
    height: u32,
    options: Vec<String>,
}

impl Draw for SelectBox {
    fn draw(&self) {
        println!("Draw a selectBox");
    }
}

fn main() {
    let screen = Screen {
        components: vec![
            Box::new(SelectBox {
                width: 75,
                height: 10,
                options: vec![
                    String::from("Yes"),
                    String::from("Maybe"),
                    String::from("No")
                ],
            }),
            Box::new(Button {
                width: 50,
                height: 10,
                label: String::from("Ok"),
            }),
        ],
    };
    
    screen.run();
}