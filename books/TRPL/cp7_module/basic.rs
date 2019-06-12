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

fn main() {
    // relative called
    sound::instrument::woodwind::clarinet();
    // absolute called
    crate::sound::voice::human();
}

// rust can define module using code only
// which is similar to java but not python