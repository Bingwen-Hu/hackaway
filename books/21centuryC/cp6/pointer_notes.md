C code function pointer
```
double a_fn(int, int); // a declaration

double (*a_fn_type)(int, int); // a function pointer

typedef double (*a_fn_type)(int, int); // function pointer type

// used in code
double apply_a_fn(a_fn_type f, int first_in, int second_in){
    return f(first_in, second_in);     
}
```
