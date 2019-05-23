struct Context<'s>(&'s str);

// lifetime subtype: s is a lifetime subtype of c
struct Parser<'c, 's: 'c> {
    // note here ---------| a lifetime is needed
    //                    |
    context: &'c Context<'s>,
}


impl<'c, 's> Parser<'c, 's> {
    fn parse(&self) -> Result<(), &'s str> {
        Err(&self.context.0[1..])
    }
}

fn parse_context(context: Context) -> Result<(), &str> {
    Parser { context: &context }.parse()
}

fn main() {
    
}