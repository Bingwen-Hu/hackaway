#pragma once

// syntax:
// ESC [ num m
// ESC [ num; num m
// ESC [ num; num; num m

// open bracket => OB => [
// semicolon => SE => ;

#define COLOR_ESC     "\033"
#define COLOR_OB      "["
#define COLOR_SE      ";"
#define COLOR_CMD     "m"
#define COLOR_STD     "0"
#define COLOR_BOLD    "1"
#define COLOR_DIM     "2"
#define COLOR_RESET   COLOR_ESC"[0m"


#define FG_BLACK     "30"
#define FG_RED       "31"
#define FG_GREEN     "32"
#define FG_YELLOW    "33"
#define FG_BLUE      "34"
#define FG_MAGENTA   "35"
#define FG_CYAN      "36"
#define FG_WHITE     "37"
#define FG_RESET     "39"
#define FG_DEFAULT   FG_RESET

#define BG_BLACK     "40"
#define BG_RED       "41"
#define BG_GREEN     "42"
#define BG_YELLOW    "43"
#define BG_BLUE      "44"
#define BG_MAGENTA   "45"
#define BG_CYAN      "46"
#define BG_WHITE     "47"
#define BG_RESET     "49"
#define BG_DEFAULT   BG_RESET

#define cfprintf1(stream, param, ...) \
    fprintf(stream, COLOR_ESC COLOR_OB param COLOR_CMD __VA_ARGS__ COLOR_RESET)

#define cprintf1(param, ...) \
    cfprintf1(stdout, param, __VA_ARGS__)

#define cfprintf2(stream, param1, param2, ...)                          \
    fprintf(stream, COLOR_ESC COLOR_OB param1 COLOR_SE param2 COLOR_CMD \
        __VA_ARGS__ COLOR_RESET)

#define cprintf2(param1, param2, ...) \
    cfprintf2(stdout, param1, param2, __VA_ARGS__)

#define cfprintf(stream, mode, fg, bg, ...)                                   \
    fprintf(stream, COLOR_ESC COLOR_OB mode COLOR_SE fg COLOR_SE bg COLOR_CMD \
        __VA_ARGS__ COLOR_RESET)

#define cprintf(mode, fg, bg, ...) \
    cfprintf(stdout, mode, fg, bg, __VA_ARGS__)


#define color_fbegin(stream, mode, fg, bg) \
    fprintf(stream, COLOR_ESC COLOR_OB mode COLOR_SE fg COLOR_SE bg COLOR_CMD)

#define color_fend(stream) \
    fprintf(stream, COLOR_RESET)

#define color_begin(mode, fg, bg) \
    color_fbegin(stdout, mode, fg, bg)

#define color_end() \
    color_fend(stdout)