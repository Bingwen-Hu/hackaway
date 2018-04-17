use std::ops::Deref;

#[derive(Debug)]
struct Mp3 {
    audio: Vec<u8>,
    artist: Option<String>,
    title: Option<String>,
}

// deref -> *
impl Deref for Mp3 {
    type Target = Vec<u8>;
    // not that return a reference
    fn deref(&self) -> &Vec<u8> {
        &self.audio
    }
}

fn main() {
    let my_favorite_song = Mp3 {
        audio: vec![1, 2, 3],
        artist: Some(String::from("Nirvana")),
        title: Some(String::from("Smells like Teen Spirit")),
    };
    assert_eq!(vec![1,2,3], *my_favorite_song);
}