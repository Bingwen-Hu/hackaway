// optioanl arguments
#include <stdio.h>
#define Testclaim(assertion, returnval) if (!(assertion))   \
    {fprintf(stderr, #assertion " failed to be true. \
    Returning " #returnval "\n"); return returnval;}


int do_things(){
    int x, y;
    x = y = 1;

    Testclaim(x==--y, -1);

    return 0;
}

void do_other_things(){
    int x, y;
    x = 1; 
    y = 2;

    Testclaim(x == y, );
    return;
}

// default arguments
#define Blankcheck(a) {int aval = (#a[0]=='\0')?2:(a+0); \
    printf("I understand your input to be %i.\n", aval); \
    }


int main(){
    int ret = do_things();
    printf("return %d\n", ret);
    do_other_things();

    Blankcheck(0);
    Blankcheck();
}
