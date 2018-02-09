// library2 file io

#include <stdio.h>
#include <stdlib.h>
int main(int argc, char* argv[argc+1]){
    if (!freopen("mylog.txt", "a", stdout)){
        perror("freopen failed");
        return EXIT_FAILURE;
    }
    puts("feeling fine today");
    return EXIT_SUCCESS;

}


/** Test Note:
stdout is bind to a file "mylog.txt", so `puts` put message into "mylog.txt"
*/
