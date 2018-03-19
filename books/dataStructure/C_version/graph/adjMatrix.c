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

            // seek the relation of ith vert
            while (!QueueEmpty(&Q)){
                DeQueue(&Q, &i);
                // check for rest element on that vert
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
    DFSTraverse(m);
    BFSTraverse(m);
    MiniSpanTree_Prim(&m);
}
