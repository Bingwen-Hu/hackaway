#pragma once

typedef char VertexType;
typedef int EdgeType;
#define MAXVEX 100
#define INFINITY 1024

typedef struct {
    VertexType vexs[MAXVEX];        // point table
    EdgeType arc[MAXVEX][MAXVEX];   // relation table
    int numVertexes, numEdges;
} MGraph;

void CreateMGraph(MGraph *G);
void printGraph(MGraph *G);