CC=c++
OBJS=ADPaper AB shifts ABBA

clean: 
	rm -rf *.o
	rm -rf $(OBJS)


ADPaper: ADPaper.o
	$(CC) ADPaper.cpp -o $@

AB: AB.o
	$(CC) AB.cpp -o $@

ABBA: ABBA.o
	$(CC) ABBA.cpp -o $@

shifts: shifts.o
	$(CC) shifts.cpp -o $@