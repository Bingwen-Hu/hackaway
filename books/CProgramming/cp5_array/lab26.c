/* judge whether an array is palindrome */


#include <stdio.h>


void isPalindrome(int lst[], int len){
  int flag = 1;
  for (int i=0, j=len-1; i<len/2; i++, j--){
    if (lst[i]!=lst[j]){
      flag = 0;
      break;
    }
  }
  if (flag)
    puts("Yes, palindrome!");
  else
    puts("No, not palindrome!");
}


int main(){

  int lst9[] = {1, 1, 2, 3, 6, 3, 2, 1, 1};
  int lst9_[] = {1, 1, 2, 3, 6, 3, 3, 1, 1};
  
  int lst8[] = {1, 2, 3, 4, 4, 3, 2, 1};
  int lst8_[] = {1, 2, 3, 4, 4, 2, 2, 1};
  
  isPalindrome(lst9, 9);
  isPalindrome(lst9_, 9);
  isPalindrome(lst8, 8);
  isPalindrome(lst8_, 8);
  

  return 0;
}

/* Test Note:
   
   I use two iteration variable one times

 */
