#include <stdio.h>
#include "array.h"


int main() {
   double *v; 
   int n = 5;
   make_vector(v, n); 
   for (int i = 0; i < n; i++) {
       v[i] = 1.0 / (1 + i);
   } 
   print_vector("%7.3f", v, n);
   free_vector(v);
}


