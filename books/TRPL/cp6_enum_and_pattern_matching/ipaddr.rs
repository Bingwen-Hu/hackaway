#[derive(Debug)]
enum IpAddrKind {
    V4,
    V6,
}

fn  main() {
    basic();
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