#[derive(Debug)]
enum IpAddrKind {
    V4,
    V6,
}

fn  main() {
    basic();
    bound_value();
    enum_with_value();
}

fn basic() {
    let four = IpAddrKind::V4;
    let six = IpAddrKind::V6;

    fn route(ip_type: &IpAddrKind) {
        match ip_type {
            IpAddrKind::V4 => println!("V4 version!"),
            IpAddrKind::V6 => println!("V6 version!"),
        };
    }    
    route(&four);
    route(&six);
    println!("{:?}", six);
}

// basic value bound 
fn bound_value() {
    #[derive(Debug)]
    struct IpAddr {
        kind: IpAddrKind,
        address: String,
    }

    let home = IpAddr {
        kind: IpAddrKind::V4,
        address: String::from("127.0.0.1"),
    };
    
    let loopback = IpAddr {
        kind: IpAddrKind::V6,
        address: String::from("::1"),
    };

    println!("home {:?}", home);
    println!("loopback {:?}", loopback);
}

fn enum_with_value() {
    #[derive(Debug)]
    enum IpAddr {
        V4(u8, u8, u8, u8),
        V6(String),
    }

    let home = IpAddr::V4(127, 0, 0, 1);
    let loopback = IpAddr::V6(String::from("::1"));

    println!("home {:?}", home);
    println!("loopback {:?}", loopback);
}