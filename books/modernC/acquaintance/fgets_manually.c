// implement fgets
// Rule 1.8.2.11 End of file can be detected after a failed read

#include <stdio.h>

char* fgets_manully(char s[restrict], int n,
                    FILE* restrict stream) {
    if (!stream) return 0;
    if (!n) return s;

    for (size_t i = 0; i < n-1; ++i){
        int val = fgetc(stream);
        switch (val) {
        case EOF:
            if (feof(stream)){
                s[i] = 0;
                return s;
            } else {
                return 0;
            }

            case '\n': s[i] = val; s[i+1] = 0; return s;
            default: s[i] = val;
        }
    }
    s[n-1] = 0;
    return s;
}


int main(){
    char string[100];
    fgets_manully(string, 99, stdin);
    puts(string);

    return 0;
}
