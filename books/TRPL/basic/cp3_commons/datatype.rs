fn main() {
    
    // here, if u32 is omited, rust compile will not able to figure out 
    // what type guess is and will cause a compile error
    let guess: u32 = "42".parse().expect("Not a number!");
    
    println!("What I guess is {}", guess);


    test_tuple();
    test_array();
}


fn test_tuple(){
    let tup = (500, 5.3, "Good");

    println!("The first element of tuple is: {}", tup.1);
}


fn test_array(){
    const K: usize = 2;
    let _array = [1, 2, 3, 4, 5];
    let array: [i32; 5] = [11, 12, 13, 14, 15];
    let _matrix: [[i32; K]; 3] = [[1, 2], [3, 4], [5, 6]];
    let another = [3; 10];

    println!("The fifth element of array is {}", array[4]);
    println!("The array {:?}", another);
}

// so tuple can mix any type of data and access as tuple.index
// array can only contains the some type of data and access as array[index]
