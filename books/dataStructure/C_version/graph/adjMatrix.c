/** Adjacency Matrix represent graph
 */
#include <stdio.h>
#include "queue.h"

typedef char VertexType;
typedef int EdgeType;
#define MAXVEX 100
#define INFINITY 1024

typedef struct {
    VertexType vexs[MAXVEX];        // point table
    EdgeType arc[MAXVEX][MAXVEX];   // relation table
    int numVertexes, numEdges;
} MGraph;

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

/* Depth first search using adjacent Matrix */
typedef enum {FALSE, TRUE} Boolean;
Boolean visited[MAXVEX];

// DFS
// @param i: index of one point
void DFS(MGraph G, int i){
    visited[i] = TRUE;
    // print vex
    printf("%c ", G.vexs[i]);
    
    // scan the ith row, if there is some point
    // unvisited, go along with the point, so 
    // we go deeper and deeper
    for (int j = 0; j < G.numVertexes; j++){
        if (G.arc[i][j] == 1 && !visited[j]) 
            DFS(G, j);
    }
}

void DFSTraverse(MGraph G){
    // every point is not visited at the beginning
    for (int i=0; i < G.numVertexes; i++){
        visited[i] = FALSE;
    }

    // for every point, if it is not visited, using DFS
    // at that point
    for (int i=0; i < G.numVertexes; i++){
        if (!visited[i]){
            DFS(G, i); 
        }
    }
    puts("");
}

void BFSTraverse(MGraph G){
    // init
    Queue Q;
    InitQueue(&Q);
    for (int i=0; i < G.numVertexes; i++){
        visited[i] = FALSE;
    }
    
    for (int i=0; i < G.numVertexes; i++){
        if (!visited[i]){
            visited[i] = TRUE;
            printf("%c ", G.vexs[i]);
            EnQueue(&Q, i);
            while (!QueueEmpty(&Q)){
                DeQueue(&Q, &i);
                for (int j=0; j < G.numVertexes; j++){
                    if (G.arc[i][j] == 1 && !visited[j]){
                        visited[j] = TRUE;
                        printf("%c ", G.vexs[j]);
                        EnQueue(&Q, j);
                    }
                }
            }
        }
    }
    puts("");
}


int main(){
    MGraph m = {.arc = {0}};
    CreateMGraph(&m);
    printGraph(&m);
    DFSTraverse(m);
    BFSTraverse(m);
}
