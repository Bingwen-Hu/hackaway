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
