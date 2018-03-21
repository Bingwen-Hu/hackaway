/* when I say compound I means list, tuple, array and so and so on */

fn show_tuple(){
    println!("show tuple!");
    let tuple: (i32, f64, char) = (42, 3.14, 'A');
    // Warning: Rust love snake case name?
    let (secret, pi, a) = tuple;
    
    println!("{} {} {} ", secret, pi, a);
    println!("{} {} {} ", tuple.0, tuple.1, tuple.2);
}

/* array in Rust is similar to C */
fn show_array(){
    println!("Show array!");
    let a = [1, 2, 4, 5, 7, 8];
    
    println!("First value of a is {}", a[0]);
}

fn main(){
    show_tuple();
    show_array();
}