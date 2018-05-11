/* tower of hanoi C implement 

move n rings from A to C using B as space

  A           B          C
  #           |          |
 ###          |          |
#####         |          |
--------------------------------


  A          B           C
  |          |           #
  |          |          ###
  |          |         #####
--------------------------------


*/
#include <stdio.h>


void hanoi_tower(int n, char src, char dest, char spare) {
    if (n == 1) {
        printf("%c --> %c\n", src, dest);
    } else {
        hanoi_tower(n - 1, src, spare, dest);
        hanoi_tower(1, src, dest, spare);
        hanoi_tower(n - 1, spare, dest, src);
    }
}

int main() {
    hanoi_tower(5, 'A', 'C', 'B');
}