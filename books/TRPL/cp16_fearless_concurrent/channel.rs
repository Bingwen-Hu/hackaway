// a channel comtains two parts:
// a transmitter and a receiver


// mspc stands for 
// multiple producer, single consumer
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn basic() {
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let val = String::from("hi");
        tx.send(val).unwrap();
    });
    /* see also try_recv */
    let received = rx.recv().unwrap();
    println!("Got: {}", received);
}

fn send_more() {
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let vals = vec![
            String::from("Hi"),
            String::from("From"),
            String::from("the"),
            String::from("thread"),
        ];

        for val in vals {
            tx.send(val).unwrap();
            thread::sleep(Duration::from_secs(1));
        }
    });

    for received in rx {
        println!("Got: {}", received);
    }
}

fn more_sender() {
    let (tx, rx) = mpsc::channel();
    let tx_cp = mpsc::Sender::clone(&tx);
    thread::spawn(move || {
        let vals = vec![
            String::from("Hi"),
            String::from("From"),
            String::from("the"),
            String::from("thread"),
        ];

        for val in vals {
            tx.send(val).unwrap();
            thread::sleep(Duration::from_secs(1));
        }
    });

    thread::spawn(move || {
        let vals = vec![
            String::from("More"),
            String::from("Message"),
            String::from("For"),
            String::from("You"),
        ];

        for val in vals {
            tx_cp.send(val).unwrap();
            thread::sleep(Duration::from_secs(1));
        }
    });


    for received in rx {
        println!("Got: {}", received);
    }
}

fn main() {
    more_sender();   
}