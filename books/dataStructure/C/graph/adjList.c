/** adjList for Graph
 */
#include "adjList.h"
#include <stdio.h>
#include <stdlib.h>
#define MAXVEX 100

void createALGraph(GraphAdjList *G){
    EdgeNode *e;
    printf("Input count of points and edges: (points,edges)\n");
    scanf("%d,%d", &G->numVertexes, &G->numEdges);
    printf("Points: %d\nEdges: %d\n", G->numVertexes, G->numEdges);
    
    getchar();
    printf("Points initialization, please input %d chars:\n", 
            G->numVertexes);
    for (int i=0; i < G->numVertexes; i++){
       G->adjList[i].data = getchar();  
       G->adjList[i].firstedge=NULL;
    }    
    getchar(); // discard \n

    int i, j;
    printf("Edges initialization: \n");
    for (int k=0; k < G->numEdges; k++){
        printf("enter subscript of edge vi, vj: \n"); 
        scanf("%d,%d", &i, &j);

        // note: for a non-direction graph, we can insert i and j
        // at the same time.
        e = malloc(sizeof(EdgeNode));
        e->adjvex = j;
        e->next=G->adjList[i].firstedge;
        G->adjList[i].firstedge=e;

        e = malloc(sizeof(EdgeNode));
        e->adjvex = i;
        e->next=G->adjList[j].firstedge;
        G->adjList[j].firstedge=e;
    }
}


void printGraphAdjlist(GraphAdjList *G){
    printf("GraphAdjlist information:\n");
    printf("-------------------------\n");
    EdgeNode *e; 
    for (int i=0; i < G->numVertexes; i++){
        printf("Point %c: ", G->adjList[i].data);
        e = G->adjList[i].firstedge; 
        while (e != NULL){
            printf("%3d", e->adjvex); 
            e = e->next;
        }
        puts("");
    }
}
