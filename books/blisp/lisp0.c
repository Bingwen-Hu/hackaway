#include <stdio.h>

static char input[2048];


int main(int argc, char** argv) {
  

  /* Print version and Exit Information. */
  puts("Lispy Version 0.0.0.0.1");
  puts("Press Ctrl+c to Exit\n");

  /* In a never ending loop */
  while(1) {

    /* Output our prompt */
    fputs("lispy> ", stdout);

    /* Read a line of user input */
    fgets(input, 2048, stdin);


    /* Echo input back to user. */
    printf("No you're a %s", input);
  }

  return 0;
}
