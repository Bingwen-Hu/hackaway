### data format


    C           Intel-data-type   Assemble  size
    *********************************************
    char        byte                b        1
    short       word                w        2
    int         double word         l        4
    long        quad word           q        8
    char*       quad word           q        8
    float       single precision    s        4
    double      double precision    l        8



### integer registers

     %rax 64-bit  quad word
     %eax 32-bit  double word
     %ax  16-bit  word
     %al   8-bit  byte


### operand specifiers

     Type        Form            Operand-value         Name
     *********************************************************************
     Immediate   $Imm             Imm                   Immediate
     Register    r                R[r]                  Register
     Memory      Imm              M[Imm]                Absolute
     Memory      (r)              M[R[r]]               Indirect
     Memory      Imm(r)           M[Imm+R[r]]           Base+displacement
     Memory      (rb, ri)         M[R[rb]+R[ri]]        Indexed
     Memory      Imm(rb, ri)      M[Imm+R[rb]+R[ri]]    Indexed
     Memory      (, ri, s)        M[R[ri]*s]            Scaled Indexed
     Memory      Imm(, ri, s)     M[Imm+R[ri]*s]        Scaled Indexed
     Memory      (rb, ri, s)      M[R[rb]+R[ri]*s]      Scaled Indexed
     Memory      Imm(rb, ri, s)   M[Imm+R[rb]+R[ri]*s]  Scaled Indexed

### instructions

    Instruction        Effect                         Description
    *********************************************************************
    pushq S            R[%rsp] <- R[%rsp]-8;      Push quad word
                       M[R[%rsp]] <- S
    popq D             D <- M[R[%rsp]];           Pop quad word
                       R[%rsp] <- R[%rsp]+8
    leaq S, D          D <- &S                    Load effectivea address

    inc D              D <- D+1                   Increment
    dec D              D <- D-1                   Decrement
    neg D              D <- -D                    Negate
    not D              D <- ~D                    Complement

    add S, D           D <- D+S                   Add
    sub S, D           D <- D-S                   subtract
    imul S, D          D <- D*S                   multiply
    xor S, D           D <- D^S                   Exclusive-or
    or S, D            D <- D|S                   Or
    and S, D           D <- D&S                   And

    sal k, D           D <- D<<k                  left shift
    shl k, D           D <- D<<k                  left shift(same as sal)
    sar k, D           D <- D>>k                  arithmetic right shift
    shr k, D           D <- D>>k                  logical right shift
    
    
