CC=gcc
CFLAGS=-g -Wall -std=c11
OBJECTS=DFS_BFS_Matrix adjList prim_matrix kruskal dijkstra
OBJECTS+=floyd

clean:
	rm -rf *.o
	rm -rf $(OBJECTS)

DFS_BFS_Matrix: DFS_BFS_Matrix.o queue.o adjMatrix.o
	$(CC) DFS_BFS_Matrix.c queue.c adjMatrix.c $(CFLAGS) -o $@

prim_matrix: prim_matrix.o queue.o adjMatrix.o
	$(CC) prim_matrix.c queue.c adjMatrix.c $(CFLAGS) -o $@

adjList: adjList.o queue.o
	$(CC) adjList.c queue.c $(CFLAGS) -o $@

kruskal: kruskal.o
	$(CC) kruskal.c $(CFLAGS) -o $@

dijkstra: dijkstra.o adjMatrix.o
	$(CC) dijkstra.c adjMatrix.c $(CFLAGS) -o $@

floyd: floyd.o adjMatrix.o
	$(CC) floyd.c adjMatrix.c $(CFLAGS) -o $@