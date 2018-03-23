// yet another mini span tree algorithms
// Kruskal algorithms focus on edges rather than points
// basic
// 1. sort all the edges according to weights, lowest first.
// 2. check every edges, if not circle is generated, add it to spam tree
// 3. done.


#include <stdio.h>

#define MAXEDGE 100
#define MAXVEX 100
// define an edges array to represent
typedef struct Edge{
    int begin;
    int end;
    int weight;
} Edge;

int find(int *parent, int f);
void MiniSpanTree_Kruskal();


void MiniSpanTree_Kruskal(){
    int numEdges = 15;
    // int numVertexes = 9;
    int n, m;
    Edge edges[MAXEDGE] = {{4, 7, 7},
                           {2, 8, 8},
                           {0, 1, 10},
                           {0, 5, 11},
                           {1, 8, 12},
                           {3, 7, 16}, 
                           {1, 6, 16}, 
                           {5, 6, 17},
                           {1, 2, 18}, 
                           {6, 7, 19},
                           {3, 4, 20},
                           {3, 8, 21},
                           {2, 3, 22},
                           {3, 6, 24},
                           {4, 5, 26}};
    int parent[MAXVEX] = {0};
    
    for (int i = 0; i < numEdges; i++){
        n = find(parent, edges[i].begin);
        m = find(parent, edges[i].end);

        // n == m means a circle has been generated
        if (n != m) {
            parent[n] = m;
            printf("(%d, %d) %d\n", edges[i].begin, 
                    edges[i].end, edges[i].weight);
        }
    }
}

int find(int *parent, int f){
    while (parent[f] > 0) {
        f = parent[f];
    }
    return f;
}


int main(){
    MiniSpanTree_Kruskal();
}