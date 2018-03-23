#pragma once


#define MAXVEX 100

typedef char VertexType;
typedef int EdgeType;

typedef struct EdgeNode {
    int adjvex;                 // store subscrpit of points
    EdgeType weight;            // store weight for network graph
    struct EdgeNode *next;
} EdgeNode;

typedef struct VertexNode {
    int in;                     // in-degree
    VertexType data;            // point field
    EdgeNode *firstedge;        // head pointer of edgelist
} VertexNode, AdjList[MAXVEX];


typedef struct {
    AdjList adjList;
    int numVertexes, numEdges;  // points and edges number
} GraphAdjList;


void createALGraph(GraphAdjList *G);
void printGraphAdjlist(GraphAdjList *G);