// array
// Rule 1.6.1.2 An array in a condition evaluates to true
// Rule 1.6.1.3 There are array objects but no array values
// Rule 1.6.1.4 Arrays can't be compared
// Rule 1.6.1.5 Arrays can't be assigned to


// FLA: Fix length array
// VLA: Variable length array

// Rule 1.6.1.6 VLA can't have initializers
// Rule 1.6.1.7 VLA can't be declared outside functions


// Rule 1.6.1.11 The length of an array A is (sizeof A) / (sizeofA[0])
// Rule 1.6.1.12 The innermost dimension of an array parameter to a function is lost
// Rule 1.6.1.13 Don't use the sizeof operator on array parameters to functions

// Rule 1.6.1.14 Array parameters behave as-if the array is passed by reference
#include <stdio.h>
#include <stdlib.h>


/* this function signature is readable*/
void swap_double(double a[static 2]){
    double tmp = a[0];
    a[0] = a[1];
    a[1] = tmp;
}

int main(void){
    double A[2] = {1.0, 2.0, };
    swap_double(A);
    printf("A[0] = %g, A[1] = %g\n", A[0], A[1]);

    return EXIT_SUCCESS;
}


