CC=c++
OBJS=ADPaper AB shifts

clean: 
	rm -rf *.o
	rm -rf $(OBJS)


ADPaper: ADPaper.o
	$(CC) ADPaper.cpp -o $@

AB: AB.o
	$(CC) AB.cpp -o $@

shifts: shifts.o
	$(CC) shifts.cpp -o $@