/** strtoul.c
provide a numerical value for all alphanumerical character
*/


#include <string.h>
#include <stdio.h>
_Static_assert('z'-'a' == 25,
               "alphabetic characters not contiguous");

#include <ctype.h>
/* convert an alphanumeric digit to an unsigned */
/* '0' ... '9'  =>  0 ..  9u */
/* 'A' ... 'Z'  => 10 .. 35u */
/* 'a' ... 'z'  => 10 .. 35u */
/* other values =>   greater */
unsigned hexatridecimal(int a){
    if (isdigit(a)){
        return a - '0';
    } else {
        a = toupper(a);
        return (isupper(a)) ? 10 +(a - 'A') : -1;
    }
}


void main(){
    int d = 9;
    char c = 'A';
    char c2 = 'a';

    printf("hex of 9 is %u", hexatridecimal(d));
    printf("hex of A is %u", hexatridecimal(c));
    printf("hex of a is %u", hexatridecimal(c2));

}


/** Test Note
It seems crashed
*/
