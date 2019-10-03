#include <stdio.h>
#include <stdlib.h>


#define TRUE 1
#define FALSE 0


int inArray(int key, int* array, int arraySize){
    for (int i=0; i < arraySize; i++) {
        if (key == array[i]) {
            return TRUE;
        }
    } 
    return FALSE;
}
