#include <stdlib.h>
#include <stdio.h>

/* code... */
huffman(int A[], int n){
    int Q[] = C;
    for(int i=0; i<n-1; i++) {
        allocate a new node z;
        z.left = x = extract-min(Q); /* pop x */
        z.right = y = extract-min(Q); /* pop y */
        z.freq = x.freq + y.freq;
        insert(Q, z);
    }
    return extract-min(Q);      /* only one tree */
}
