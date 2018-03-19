#include "adjMatrix.h"
#include "queue.h"
#include <stdio.h>


// Mini Span Tree algorithms -- Prim
// basic: 
// start from any point, and search for every edge connect to it
// find the lowest cost point and add it to span tree
// then search on the new accepted point and using its weights
// to override the cost value if lower.
void MiniSpanTree_Prim(MGraph *G){
    int min;
    int adjvex[MAXVEX];
    int lowcost[MAXVEX];
    
    // set v0 as 0, if lowcost[i] == 0 then it means
    // i is in span tree
    lowcost[0] = 0;
    adjvex[0] = 0;                          
    // initalization
    for (int i=1; i < G->numVertexes; i++){ // except v0
        lowcost[i] = G->arc[0][i];
        adjvex[i] = 0;                      // init as 0
    }

    for (int i=1; i < G->numVertexes; i++){
        min = INFINITY;
        int j = 1, k = 0;
        while (j < G->numVertexes){
            if (lowcost[j] != 0 && lowcost[j] < min){
                min = lowcost[j];
                k = j;
            }
            j++;
        }

        printf("(%d, %d)%s", adjvex[k], k, i == G->numVertexes-1 ? "\n" : " ");   // print current lowest weight
        lowcost[k] = 0;                     // accept the point

        for (j = 1; j < G->numVertexes; j++){
            if (lowcost[j] != 0 && G->arc[k][j] < lowcost[j]){
                lowcost[j] = G->arc[k][j];
                adjvex[j] = k;
            }
        }
    }
}




int main(){
    MGraph m;
    CreateMGraph(&m);
    printGraph(&m);
    MiniSpanTree_Prim(&m);
}
