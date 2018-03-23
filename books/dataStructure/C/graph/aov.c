/** AOV algorithms 
 * If a direct graph has no circle, all the vex will print out
 */
#include "adjList.h"
#include <stdlib.h>
#include <stdio.h>

typedef enum{ERROR, OK} Status;

Status TopologicalSort(GraphAdjList *GL) {
    EdgeNode *e;
    int top = 0;
    int count = 0;
    int *stack;
    stack = (int *)malloc(GL->numVertexes * sizeof(int));
    
    for (int i=0; i < GL->numVertexes; i++) {
        if (GL->adjList[i].in == 0) {
            stack[top++] = i;
        }
    }
    while (top != 0){
        int top_ = stack[--top];
        printf("%2d %s ", GL->adjList[top_].data, top != 0 ? "->" : "\n");
        count++;
        for (e = GL->adjList[top_].firstedge; e; e = e->next) {
            int k = e->adjvex;
            if (!(--GL->adjList[k].in)) {
                stack[top++] = k;
            }
        }
    }
    
    if (count < GL->numVertexes) {
        return ERROR;
    } else {
        return OK;
    }
}

int main() {
    // TODO: initialize
}