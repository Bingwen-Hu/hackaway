// string token
#include <string.h> //strtok_r
#include <stdio.h>

int count_split(char *instring, char *delimiter){
    int counter = 0;
    char *scratch, *txt;
    while ((txt = strtok_r(!counter ? instring : NULL, delimiter, &scratch))){
        counter++;
    }
    return counter;
}




int main(){
    char string[] = "I split in blank!"; 
    char delimiter[] = " ";
    int counter = count_split(string, delimiter);
    printf("counter is %d\n", counter);
}
