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

     Type        Form             Operand-value         Name
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

    Instruction        Effect                     Description
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
    
    
### Stack frame for function

    The C code:
    long call_proc()
    {
	long x1 = 1; int x2 = 2;
	short x3 = 3; char x4 = 4;
	proc(x1, &x1, x2, &x2, x3, &x3, x4, &x4);
	return (x1+x2)*(x3-x4);
    }
    
    Generated assembly code
     1 call_proc:
     // set up arguments to proc
     2 subq    $32, %rsp        // Allocate 32-byte stack frame
     3 movq    $1, 24(%rsp)     // Store 1 in &x1
     4 movl    $2, 20(%rsp)     // Store 2 in &x2
     5 movw    $3, 18(%rsp)     // Store 3 in &x3
     6 movb    $4, 17(%rsp)     // Store 4 in &x4
     7 leaq    17(%rsp), %rax   // Create &x4
     8 movq    %rax, 8(%rsp)    // Store &x4 as argument 8
     9 movl    $4, (%rsp)       // Store 4 as argument 7
    10 leaq    18(%rsp), %r9    // Pass &x3 as argument 6
    11 movl    $3, %r8d         // Pass 3 as argument 5
    12 leaq    20(%rsp), %rcx   // Pass &x2 as argument 4
    13 movl    $2, %edx         // Pass 2 as argument 3
    14 leaq    24(%rsp), %rsi   // Pass &x1 as argument 2
    15 movl    $1, %edi         // Pass 1 as argument 1
    16 call    proc
    // retrieve changes to memory
    17 movslq  20(%rsp), %rdx   // Get x2 and convert to long
    18 addq    24(%rsp), %rdx   // Compute x1+x2
    19 movswl  18(%rsp), %eax   // Get x3 and convert to int
    20 movsbl  17(%rsp), %ecx   // Get x4 and convert to int
    21 subl    %ecx, %eax       // Compute x3-x4
    22 cltq                     // Convert to long
    23 imulq   %rdx, %rax       // Compute (x1+x2) * (x3-x4)
    24 addq    $32, %rsp        // Deallocate stack frame
    25 ret                      // Return
