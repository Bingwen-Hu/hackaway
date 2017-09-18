// have a look at for

fn main(){
    let array = [1, 2, 3, 4, 5];

    for i in array.iter() {
        println!("The element of array is: {}", i);
    }


    test_while();
}


fn test_while() {
    // no differece except the condition

    let mut i = 10;

    while i >= 0 {
        println!("I just learn the grammer: {}", i);
        i = i - 1;
    }
}
