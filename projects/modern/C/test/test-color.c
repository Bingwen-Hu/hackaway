#include <stdio.h>
#include "termcolor.h"


// how can I
// cprinf(FG_RED, "something?");


int main()
{
    // full mode
    cfprintf(stdout, COLOR_STD, FG_RED, BG_BLUE, "haha\n");
    printf(COLOR_ESC COLOR_OB FG_RED COLOR_CMD "haha\n");
    cprintf(COLOR_STD, FG_RED, BG_RESET, "haha\n");

    // one mode
    cprintf1(FG_RED, "only foreground red\n");
    cprintf1(BG_CYAN, "only background cyan\n");
    cprintf1(COLOR_BOLD, "only bold (brightness)\n");

    // two mode
    cprintf2(FG_RED, BG_BLUE, "two mode, fg_red bg_blue \n");
    cprintf2(BG_CYAN, COLOR_BOLD, "two mode, bg_cyan bold \n");
    cprintf2(COLOR_BOLD, FG_CYAN, "two mode, bold, fg_cyan\n");

    // block mode
    color_begin(COLOR_BOLD, FG_BLUE, BG_DEFAULT);
    printf("Now I'm in a block!\n");
    printf("I can have several lines\n");
    printf("After I finished, I can close it!\n");
    color_end();

    // this will failed
    // FILE* f = fopen("some.txt", "w");
    // cfprintf1(f, FG_RED, "some thing\n"); 
    // fclose(f);

    return 0;
}