#include "adjMatrix.h"
#include "queue.h"
#include <stdio.h>

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



int main(){
    MGraph m;
    CreateMGraph(&m);
    printGraph(&m);
    DFSTraverse(m);
    BFSTraverse(m);
}
