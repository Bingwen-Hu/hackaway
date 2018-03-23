#include "adjList.h"
#include "queue.h"
#include <stdio.h>

typedef enum {FALSE, TRUE} Boolean;
Boolean visited[MAXVEX];

// DFS using adjList
void DFS(GraphAdjList *GL, int i){
    EdgeNode *p;
    visited[i] = TRUE;
    printf("%c ", GL->adjList[i].data);
    p = GL->adjList[i].firstedge;
    while (p) {
        if (!visited[p->adjvex]){
            DFS(GL, p->adjvex);
        }
        p = p->next;
    }
}

void DFSTraverse(GraphAdjList *GL){
    printf("DFS: ");
    for (int i=0; i < GL->numVertexes; i++){
        visited[i] = FALSE;
    }
    for (int i=0; i < GL->numVertexes; i++){
        if (!visited[i])
            DFS(GL, i);
    }
    putchar('\n');
}

void BFSTraverse(GraphAdjList *GL){
    printf("BFS: ");
    EdgeNode *p;
    Queue Q;
    for (int i=0; i < GL->numVertexes; i++){
        visited[i] = FALSE;
    }
    InitQueue(&Q);

    for (int i=0; i < GL->numVertexes; i++){
        if (!visited[i]){
            visited[i] = TRUE;
            printf("%c ", GL->adjList[i].data);
            EnQueue(&Q, i);
            while (!QueueEmpty(&Q)){
                DeQueue(&Q, &i);
                p = GL->adjList[i].firstedge;
                while (p) {
                    if (!visited[p->adjvex]) {
                        visited[p->adjvex] = TRUE;
                        printf("%c ", GL->adjList[p->adjvex].data);
                        EnQueue(&Q, p->adjvex);
                    }
                    p = p->next;
                }
            }
        }
    }
    putchar('\n');
}


int main(){
    GraphAdjList G;
    createALGraph(&G);
    printGraphAdjlist(&G);
    DFSTraverse(&G);
    BFSTraverse(&G);
}
