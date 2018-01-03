// function pointer test
#include <stdio.h>


double max(double a, double b){
    return a>b ? a : b; 
}

double min(double a, double b){
    return a<b ? a : b;
}

typedef double (*mory_type)(double, double);

double apply_mory_type(mory_type mory_fun, double a, double b){
    return mory_fun(a, b);
}

int main(){
    double a = 20.;
    double b = 14.;
    
    double min_ = apply_mory_type(min, a, b);
    double max_ = apply_mory_type(max, a, b);
    printf("maximum of %0.lf and %0.lf is %0.lf\n", a, b, max_);
    printf("minimum of %0.lf and %0.lf is %0.lf\n", a, b, min_);
}
