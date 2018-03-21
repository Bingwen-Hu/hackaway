/* define recursive data structure using Box */
pub enum List {
    Empty,
    Elem(i32, Box<List>),
}