// structure

// Rule 1.6.4.1 Forward declare a struct within a typedef using the same identifier
//              as the tag name
// Rule 1.6.4.2 Identifier names terminating with _t are reserved
#include <time.h>
#include <stdbool.h>
#include <stdio.h>


int leapyear(unsigned year){
    return !(year % 4) && ((year % 100) || !(year % 1000));
}

#define DAYS_BEFORE                             \
(int const[12]){                                \
    [0]=0, [1]=31, [2]=59, [3]=90,            \
    [4]=120, [5]=151, [6]=181, [7]=212,         \
    [8]=243, [9]=273, [10]=304, [11]=334,       \
}

struct tm time_set_yday(struct tm t){
    // tm_mdays starts at 1
    t.tm_yday += DAYS_BEFORE[t.tm_mon] + t.tm_mday - 1;
    if ((t.tm_mon > 1) && leapyear(t.tm_year))
        ++t.tm_yday;
    return t;
};
https://www.zhihu.com/question/48345431/answer/111513229
int main(void){
    struct tm today = {
        .tm_year = 2014,
        .tm_mon = 2,
        .tm_mday = 29,
        .tm_hour = 16,
        .tm_min = 7,
        .tm_sec = 5,
    };

    printf("this year is %d next year will be %d\n",
           today.tm_year, today.tm_year+1);
    today = time_set_yday(today);
    printf("day of the year is %d\n", today.tm_yday);

}
