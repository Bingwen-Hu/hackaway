/** Any variant which is valid as a struct is also valid as an enum.
 * 
 */

#[allow(dead_code)]

enum WebEvent {
    // 'unit-like' member
    PageLoad,
    PageUnload,
    // tuple-like member
    KeyPress(char),
    Paste(String),
    // struct-like member
    Click { x: i64, y: i64},
}

fn inspect(event: WebEvent) {
    match event {
        WebEvent::PageLoad => println!("page loaded"),
        WebEvent::PageUnload => println!("page unloaded"),
        // destructure `c` from inside the `enum`
        WebEvent::KeyPress(c) => println!("pressed `{}`.", c),
        WebEvent::Paste(s) => println!("pasted \"{}\".", s),
        // destructure `Click` into `x` and `y`
        WebEvent::Click { x, y } => {
            println!("clicked at x={}, y={}.", x, y);
        },
    }
}


fn main() {
    let pressed  = WebEvent::KeyPress('x');
    // to_owned created an owned string from a string slice
    let pasted   = WebEvent::Paste("My text".to_owned());
    let click    = WebEvent::Click { x: 20, y: 30 };
    let load     = WebEvent::PageLoad;
    let unloaded = WebEvent::PageUnload;

    inspect(pressed);
    inspect(pasted);
    inspect(click);
    inspect(load);
    inspect(unloaded);
}