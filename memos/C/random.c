#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void random_int(int *first, int *second) {
    srand(time(NULL));  // 初始化随机种子，使得每次生成的随机数不一样
    *first = rand() % 10 + 1;
    *second = rand() % 10 + 1;
}

int main() {
    int first, second;

    random_int(&first, &second);

    printf("1到10的随机数: %d, %d\n", first, second);
}