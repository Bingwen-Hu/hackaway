# include <stdio.h>


int main(){
  
  char c;

  printf("Please input one character: ");
  scanf("%c", &c);

  if (c >= 'A' && c <= 'Z')
    printf("A capital! %c", c);
  else if (c >= 'a' && c < 'z')
    printf("A small case letter! %c", c);
  else if (c >= '0' && c <= '9')
    printf("A number! %c", c);
  else
    printf("c is %c", c);

  return 0;
}



/*  TEst Note

    char can be compared to each other,
    there is no need to remember the ascii code for chars

 */
