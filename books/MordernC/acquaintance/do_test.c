#include <stdio.h>
#include <tgmath.h>

void main(){
    double const eps = 1E-3;
    double const a = 34.0;
    double x = 0.5;

    while (fabs(1.0 - a*x) >= eps) {
        x *= (2.0 - a*x);
        printf("value of x is %.3f\n", x);  // endless loop
    }
}
