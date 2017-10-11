/* Insert an element in 1-d sorted array 
   delete an element by index
   delete an element by value
*/

#include <stdio.h>

#define SIZE 10

void deleteByValue(int lst[], int *len, int value);
void deleteByIndex(int lst[], int* len, int index);
int insert(int lst[], int* len, int value);
void printList(int lst[], int len);

int main(){

  int lst[SIZE] = {1, 3, 5, 7, 8, 11, 22, 35, 42, 53};
  int len = SIZE;               /* initial as SIZE */
  
  puts("orginal: ");
  printList(lst, len);
  puts("delete by index 3");
  deleteByIndex(lst, &len, 3);
  printList(lst, len);
  puts("delete by index 11");
  deleteByIndex(lst, &len, 11);
  printList(lst, len);
  puts("insert value 9");
  insert(lst, &len, 9);
  printList(lst, len);
  puts("delete by value 3");
  deleteByValue(lst, &len, 3);
  printList(lst, len);
  puts("delete by value 27");
  deleteByValue(lst, &len, 27);
  printList(lst, len);


  return 0;
}



void printList(int lst[], int len){
  printf("The list: ");
  for (int i=0; i<len; i++){
    printf("%d ", lst[i]);
  }
  putchar('\n');
}


int insert(int lst[], int *len, int value){
  int i;
  if (*len == SIZE){
    puts("List is full, insertion failed.");
  } else {
    for (i=*len-1; i>=0 && lst[i]>value; i--){
        lst[i+1] = lst[i];
    }
    lst[i+1] = value;
    (*len)++;
  }
  return i;
}


void deleteByIndex(int lst[], int *len, int index){
  int length = *len;
  if (length < index || index < 0){
    puts("invalied index!");
  } else {
    for (int i=index; i<length-1; i++){
      lst[i] = lst[i+1];
    }
    (*len)--;
  }
  return;
}

void deleteByValue(int lst[], int *len, int value){
  int length = *len;
  for (int i=0; i<length; i++){
    if (lst[i] == value){
      for (int j=i; j<length-1; j++)
        lst[j] = lst[j+1];
      (*len)--;
      break;
    }
  }
  if (length == *len)
    printf("value %d not found!\n", value);
}



/* Test Note:

   (*len)++ is very different from *len++

 */
