/* more formatting */
use std::fmt::{self, Formatter, Display};

struct City {
    name: &'static str,
    lat: f32, // latitude
    lon: f32, // longitude
}

impl Display for City {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let lat_c = if self.lat == 0.0 { 'N' } else { 'S' };
        let lon_c = if self.lon == 0.0 { 'E' } else { 'W' };

        write!(f, "{:8}: {:7.3}*{} {:7.3}*{}",
               self.name, self.lat.abs(), lat_c, self.lon.abs(), lon_c)
    }
}

#[derive(Debug)]
struct Color {
    red: u8,
    green: u8,
    blue: u8,
}

fn main() {
    for city in [
        City { name: "Dublin", lat: 53.234, lon: -4.554},
        City { name: "Oslo",   lat: 13.234, lon: 14.554},
        City { name: "Genrde", lat:  3.14,  lon: 44.554},
    ].iter() {
        println!("{}", *city);
    }

    for color in [
        Color { red: 128, green: 255, blue:  99 },
        Color { red:   0, green:   1, blue: 254 },
        Color { red:   0, green:   0, blue:   0 },
    ].iter() {
        println!("{:?}", *color);
    }
}