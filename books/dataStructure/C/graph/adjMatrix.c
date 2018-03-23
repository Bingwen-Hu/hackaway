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


