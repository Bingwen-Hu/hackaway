use std::ops::Deref;

#[derive(Debug)]
struct Mp3 {
    audio: Vec<u8>,
    artist: Option<String>,
    title: Option<String>,
}

// deref -> *
impl Deref for Mp3 {
    type Target = Vec<u8>; // associated type covered in chapter19
    // not that return a reference
    // deref enable `*` operator, like __len__ enable len() in Python
    fn deref(&self) -> &Vec<u8> {
        &self.audio
    }
}

// deref coercion, good to know
// When the Deref trait is defined for the types involved, Rust will analyze 
// the types and use Deref::deref as many times as necessary to get a reference 
// to match the parameter’s type. The number of times that Deref::deref needs 
// to be inserted is resolved at compile time, so there is no runtime penalty 
// for taking advantage of deref coercion!

fn main() {
    let my_favorite_song = Mp3 {
        audio: vec![1, 2, 3],
        artist: Some(String::from("Nirvana")),
        title: Some(String::from("Smells like Teen Spirit")),
    };
    // autually, rust run this code
    // *(my_favorite_song.deref())
    assert_eq!(vec![1,2,3], *my_favorite_song);
}
