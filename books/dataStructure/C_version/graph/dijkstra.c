/** the shortest path dijkstra algorithms 
 */
#include <stdio.h>
#include "adjMatrix.h"

typedef int Pathmatrix[MAXVEX];
typedef int ShortPathTable[MAXVEX];

/* P[v] subscript of pre-point, D[v] donates the sum of shortest path from v0 to v*/
void ShortestPath_Dijkstra(MGraph *G, int v0, Pathmatrix *P, ShortPathTable *D){
    int final[MAXVEX];
    
    // initalization
    for (int v = 0; v < G->numVertexes; v++){
        final[v] = 0;
        (*D)[v] = G->arc[v0][v];
        (*P)[v] = 0;
    }
    (*D)[v0] = 0;
    final[v0] = 1;

    // main algorithms
    // loop for every point except the start one.
    // find the nearest neighbor and save its index 
    // as k, set final[k] = 1 means this point is in.
    // min store the nearest path from v0 to v.
    for (int v = 1; v < G->numVertexes; v++){
        int min = INFINITY;
        int k;
        // find the shortest path from v0 to vk
        for (int w = 0; w < G->numVertexes; w++){
            if (!final[w] && (*D)[w] < min){
                k = w;
                min = (*D)[w];  
            }
        }
        final[k] = 1;
        // check neighbors of k, and update the path
        // and index if shorter path is found
        // base on point k, find out k->w and see if 
        // k->w + min is smaller
        for (int w = 0; w < G->numVertexes; w++){
            if (!final[w] && (min + G->arc[k][w] < (*D)[w])){
                (*D)[w] = min + G->arc[k][w];
                (*P)[w] = k;
            }
        }
    }
}

void printPath(MGraph *G, Pathmatrix P){
    int before = G->numVertexes -1;
    printf("Paths: %c%s", G->vexs[before], " ");
    while (before != 0){
        before = P[before];
        printf("%c%s", G->vexs[before], " ");
    }
    putchar('\n');
}

int main(){
    // prepare graph
    MGraph G = {
        .vexs = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'},
        .arc = {
            [0][1] = 1, [0][2] = 5, 
            [1][0] = 1, [1][2] = 3, [1][3] = 7, [1][4] = 5,
            [2][0] = 5, [2][1] = 3, [2][4] = 1, [2][5] = 7,
            [3][1] = 7, [3][4] = 2, [3][6] = 3,
            [4][1] = 5, [4][2] = 1, [4][3] = 2, [4][5] = 3, [4][6] = 6, [4][7] = 9,
            [5][2] = 7, [5][4] = 3, [5][7] = 5,
            [6][3] = 3, [6][4] = 6, [6][7] = 2, [6][8] = 7,
            [7][4] = 9, [7][5] = 5, [7][6] = 2, [7][8] = 4,
            [8][6] = 7, [8][7] = 4,
        },
        .numVertexes = 9,
        .numEdges = 15,
    };

    for (int i = 0; i < G.numVertexes; i++){
        for (int j = 0; j < G.numVertexes; j++){
            if (G.arc[i][j] == 0 && i != j){
                G.arc[i][j] = INFINITY;
            }
        }
    }
    printGraph(&G);

    Pathmatrix P = {0};
    ShortPathTable D = {0};
    int v0 = 0;

    ShortestPath_Dijkstra(&G, v0, &P, &D);
    printPath(&G, P);
}