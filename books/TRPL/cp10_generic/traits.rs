// trait function can have a default implementation  and can be 
// override by concrete type implement it.
pub trait Summarizable {
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
}