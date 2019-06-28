// A trait tells the Rust compiler about functionality a particular type has 
// and can share with other types. We can use traits to define shared behavior 
// in an abstract way. We can use trait bounds to specify that a generic can 
// be any type that has certain behavior.



// trait function can have a default implementation  and can be 
// override by concrete type implement it.

// summarize is a behavior that can be defined in Article and Tweet
pub trait Summarizable {
    // this define a default method
    fn summary(&self) -> String {
        String::from("(Read more...)")
    }
}

pub struct NewsArticle {
    pub headline: String,
    pub location: String,
    pub author: String,
    pub content: String,
}

impl Summarizable for NewsArticle {
    fn summary(&self) -> String {
        format!("{}, by {} ({})", self.headline, self.author, self.location)
    }
}


pub struct Tweet {
    pub username: String,
    pub content: String,
    pub reply: bool,
    pub retweet: bool,
}

impl Summarizable for Tweet {
    fn summary(&self) -> String {
        format!("{}: {}", self.username, self.content)
    }
}

pub struct Surpasser {
    pub name: String,
}

impl Summarizable for Surpasser {}

// trait bounds: when we call a function on a generic type, we can using train bounds
// to ensure that the generic type implement a specific traits
pub fn notify<T: Summarizable>(item: &T) {
    println!("Breaking news! {}", item.summary());
}

pub fn yanotify<T>(item: &T) 
    where T: Summarizable
{
    println!("Breaking news! {}", item.summary());
}

// it is so happy we have another choice
// this is a move version
pub fn yanotify2(item: impl Summarizable) {
    println!("Breaking news! {}", item.summary());
}


// return type with traits
// Returning a type that is only specified by the trait it implements 
// is especially useful in the context of closures and iterators
fn returns_summarizable() -> impl Summarizable {
    Tweet {
        username: String::from("horsebooks"),
        content: String::from("of course, as you probably already know"),
        reply: false,
        retweet: false,
    }
}

fn main() {
    let tweet = Tweet {
        username: String::from("Mory"),
        content: String::from("Mory want to be helpful"),
        reply: false,
        retweet: false,
    };

    println!("1 new tweet: {}", tweet.summary());

    let surpasser = Surpasser { name: "Ann".to_string() };
    println!("{}", surpasser.summary());

    notify(&surpasser);
    yanotify(&surpasser);
    yanotify2(surpasser); // note that surpasser move in
}