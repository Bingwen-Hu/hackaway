/* floyd algorithms calculate all the shortest path of a graph */

#include <stdio.h>
#include "adjMatrix.h"

typedef int Pathmatrix[MAXVEX][MAXVEX];
typedef int ShortPathTable[MAXVEX][MAXVEX];


void ShortestPath_Floyd(MGraph *G, Pathmatrix *P, ShortPathTable *D){
    // init
    for (int v=0; v < G->numVertexes; v++) {
        for (int w=0; w < G->numVertexes; w++) {
            (*D)[v][w] = G->arc[v][w];
            (*P)[v][w] = w;
        }
    }

    for (int k=0; k < G->numVertexes; k++) {
        for (int v=0; v < G->numVertexes; v++) {
            for (int w=0; w < G->numVertexes; w++) {
                if ((*D)[v][w] > (*D)[v][k] + (*D)[k][w]) {
                    (*D)[v][w] = (*D)[v][k] + (*D)[k][w];
                    (*P)[v][w] = (*P)[v][k];
                }
            }
        }
    }
}

int main() {
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

    ShortPathTable D;
    Pathmatrix P;

    for (int i=0; i < G.numVertexes; i++) {
        for (int j=0; j < G.numVertexes; j++) {
            P[i][j] = j;
            D[i][j] = G.arc[i][j];
        }
    }
    ShortestPath_Floyd(&G, &P, &D);

    // show it
    puts("PathMatrix");
    puts("----------");
    for (int i=0; i < G.numVertexes; i++) {
        for (int j=0; j < G.numVertexes; j++) {
            printf("%3d%s", P[i][j], j < G.numVertexes-1 ? " " : "\n");
        }
    }
    printf("\n\n");
    puts("WeightMatrix");
    puts("------------");
    for (int i=0; i < G.numVertexes; i++) {
        for (int j=0; j < G.numVertexes; j++) {
            printf("%3d%s", D[i][j], j < G.numVertexes-1 ? " " : "\n");
        }
    }
}