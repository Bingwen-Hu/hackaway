#include <stdio.h>
#include <string.h>

#define NUM 5

void main(){

  char names[NUM][9] = {"Mory", "Ann", "Jenny", "Sirius", "Demon"};
  char name_t[9];
  printf("Origin names:");
  for (int i=0; i<NUM; i++){
    printf(" %s", names[i]);
  }
  puts("");

  int j;
  for (int i=1; i<NUM; i++){
    strcpy(name_t, names[i]);
    for (j=i-1; j>=0 && strcmp(names[j], name_t)>0; j--){
      strcpy(names[j+1], names[j]);
    }
    strcpy(names[j+1], name_t);
    
    printf("Sorting %d:", i);
    for (int k=0; k<NUM; k++){
      printf(" %s", names[k]);
    }
    puts("");

  }

  printf("After sorting:");
  for (int i=0; i<NUM; i++){
    printf(" %s", names[i]);
  }
 

 


}


/* Test Note:


   strcmp: string compare function
   strcpy: copy one string to another.

   insertion sort is so sweet!
   
 */
