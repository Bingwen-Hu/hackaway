fn main() {
    let favorite_color: Option<&str> = None;
    let is_tuesday = false;
    let age: Result<u8, _> = "35".parse();

    if let Some(color) = favorite_color {
        println!("Using your favorite color, {}, as the background.", color);
    } else if is_tuesday {
        println!("Tuesday is green day!");
    } else if let Ok(age) = age {
        if age > 30 {
            println!("Using purple as background.");
        } else {
            println!("Using orange as background.");
        }
    } else {
        println!("using blue as the background color!");
    }
}