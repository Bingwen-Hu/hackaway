CC=gcc
CFLAGS=-g -Wall -std=c11
OBJECTS=DFS_BFS_Matrix adjList

clean:
	rm -rf *.o
	rm -rf $(OBJECTS)

DFS_BFS_Matrix: DFS_BFS_Matrix.o queue.o adjMatrix.o
	$(CC) DFS_BFS_Matrix.c queue.c adjMatrix.c $(CFLAGS) -o $@

adjList: adjList.o queue.o
	$(CC) adjList.c queue.c $(CFLAGS) -o $@