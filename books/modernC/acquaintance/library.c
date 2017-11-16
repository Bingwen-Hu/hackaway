/** library
C11 addressed the security problem in C libs. Bounds-checking interfaces of
Annex K is provided

#if !__STDC_LIB_EXT1__
#error "This code needs bounds checking interface Annex K"
#endif
#define __STDC_WANT_LIB_EXT1__ 1
*/

// Rule 1.8.0.4 Identifier names terminating with _s are reserved
// Rule 1.8.0.5 Missed preconditions for the execution platform must abort compilation
// Rule 1.8.0.7 In preprocessor conditions unknown identifiers evaluate to 0

// Rule 1.8.2.3 puts and fputs differ in their end of line handlinng

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[argc+1]){
    FILE* logfile = fopen("mylog.txt", "a");
    if (!logfile){
        perror("fopen_failed");
        return EXIT_FAILURE;
    }
    fputs("Feeling fine today\n", logfile);
    return EXIT_SUCCESS;
}
