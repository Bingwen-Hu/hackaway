/** Adjacency Matrix represent graph
 */
#include <stdio.h>
#include "queue.h"
#include "adjMatrix.h"

void CreateMGraph(MGraph *G){
    int i, j, k, w;
    printf("Input number of points and edges: (num_p,num_e)\n");
    scanf("%d,%d", &G->numVertexes, &G->numEdges);
    printf("Points: %d\nEdges: %d\n", G->numVertexes, G->numEdges);

    getchar();
    printf("start enter %d chars: \n", G->numVertexes);
    for (i = 0; i < G->numVertexes; i++)
        G->vexs[i] = getchar();
    getchar();
    for (i = 0; i < G->numVertexes; i++)
        for (j = 0; j < G->numVertexes; j++)
            G->arc[i][j] = INFINITY;
    
    for (k = 0; k < G->numEdges; k++){
        printf("input subscrpit i,j and weight w: (i,j,w)\n"); 
        scanf("%d,%d,%d", &i, &j, &w);
        printf("arc[%d][%d]=%d\n", i, j, w);
        G->arc[i][j] = w;
        G->arc[j][i] = G->arc[i][j];
    }
}


void printGraph(MGraph *G){
    printf("-----------------\n");
    printf("Graph information\n"); 
    printf("-----------------\n");
    for (int i=0; i < G->numVertexes; i++){
        for (int j=0; j < G->numVertexes; j++){
            printf("%6d", G->arc[i][j]);     
        }
        puts("\n");
    }
    puts("\n");
}



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
