// exercise 2.10
// exercise 2.11
#include <stdio.h>
#include <stdlib.h>

void swap(int *, int *);
void test_2_10();
void test_2_11();

int main(){
    //test_2_10();
    test_2_11();
}




void test_2_10(){
    int first=3, second=5; // 2.10
    printf("value of first and second is: %d %d\n", first, second);
    swap(&first, &second);
    printf("after swap first and second is: %i %i\n", first, second);
}

void test_2_11(){
    int *first = malloc(sizeof(int));
    int *second = malloc(sizeof(int));
    *first = 4;
    *second = 34;

    printf("value of first and second is: %d %d\n", *first, *second);
    swap(first, second);
    printf("after swap first and second is: %i %i\n", *first, *second);

    free(first);
    free(second);
}


void swap(int *first, int *second){
    int temp = *first;
    *first = *second;
    *second = temp;
}

