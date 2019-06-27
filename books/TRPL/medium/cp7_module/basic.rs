// in src/main.rs of a project
mod sound {
    pub mod instrument {
        pub mod woodwind {
            pub fn clarinet() {
                println!("sound::instrument::woodwind::clarinet() called!");
            }
        }
    }
    pub mod voice {
        pub fn human() {
            println!("Relative called!");
            super::instrument::woodwind::clarinet();
        }
    }
}

fn in_short() {
    use self::sound::instrument::woodwind;
    use self::sound::voice::human as rhuman;
    woodwind::clarinet();
    rhuman();
}

fn pub_use() {
    // check it out if you need
}

fn main() {
    // relative called
    sound::instrument::woodwind::clarinet();
    // absolute called
    crate::sound::voice::human();

    // use bring things into scope
    in_short();

    pub_use();
}

// rust can define module using code only
// which is similar to java but not python