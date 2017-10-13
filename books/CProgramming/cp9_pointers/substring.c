#include <stdio.h>
#include <string.h>

void main(){

  char str[200], news[200];
  char *s, *t;
  int pos, n, i;

  printf("Enter the string: ");
  scanf("%s", str);
  printf("Enter the position and number of characters to extract: ");
  scanf("%d %d", &pos, &n);
  
  s = str;
  t = news;
  if(n==0)
    n = strlen(str);
  s = s + pos - 1;
  for(i=0; i<n; i++){
    *t = *s;
    s++;
    t++;
  }
  *t = '\0';
  printf("The substring is: %s\n", news);

}
