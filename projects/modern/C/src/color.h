#pragma once

// syntax:
// ESC [ num m
// ESC [ num; num m
// ESC [ num; num; num m
// 

// basic code
#define ESC     "\033"
#define SEP     ";"
#define CMD     "m"
#define STD     ESC"[0;"
#define BOLD    ESC"[1;"
#define DIM     ESC"[2;"
#define RESET   ESC"[0m"

#define FG_BLACK     "30"
#define FG_RED       "31"
#define FG_GREEN     "32"
#define FG_YELLOW    "33"
#define FG_BLUE      "34"
#define FG_MAGENTA   "35"
#define FG_CYAN      "36"
#define FG_WHITE     "37"
#define FG_RESET     "39"

#define BG_BLACK     "40"
#define BG_RED       "41"
#define BG_GREEN     "42"
#define BG_YELLOW    "43"
#define BG_BLUE      "44"
#define BG_MAGENTA   "45"
#define BG_CYAN      "46"
#define BG_WHITE     "47"
#define BG_RESET     "49"


// standard mode as default
#define STD_BLACK     STD"30m"
#define STD_RED       STD"31m"
#define STD_GREEN     STD"32m"
#define STD_YELLOW    STD"33m"
#define STD_BLUE      STD"34m"
#define STD_MAGENTA   STD"35m"
#define STD_CYAN      STD"36m"
#define STD_WHITE     STD"37m"
#define STD_RESET     STD"39m"

// bold (brightness) mode
#define BOLD_BLACK     BOLD"30m"
#define BOLD_RED       BOLD"31m"
#define BOLD_GREEN     BOLD"32m"
#define BOLD_YELLOW    BOLD"33m"
#define BOLD_BLUE      BOLD"34m"
#define BOLD_MAGENTA   BOLD"35m"
#define BOLD_CYAN      BOLD"36m"
#define BOLD_WHITE     BOLD"37m"
#define BOLD_RESET     BOLD"39m"

// dim (darkness) mode
#define DIM_BLACK     DIM"30m"
#define DIM_RED       DIM"31m"
#define DIM_GREEN     DIM"32m"
#define DIM_YELLOW    DIM"33m"
#define DIM_BLUE      DIM"34m"
#define DIM_MAGENTA   DIM"35m"
#define DIM_CYAN      DIM"36m"
#define DIM_WHITE     DIM"37m"
#define DIM_RESET     DIM"39m"

// background
#define BG_STD_BLACK     STD"40m"
#define BG_STD_RED       STD"41m"
#define BG_STD_GREEN     STD"42m"
#define BG_STD_YELLOW    STD"43m"
#define BG_STD_BLUE      STD"44m"
#define BG_STD_MAGENTA   STD"45m"
#define BG_STD_CYAN      STD"46m"
#define BG_STD_WHITE     STD"47m"
#define BG_STD_RESET     STD"49m"


#define ESC_STD_RED_WHITE ESC_STD"31;47m"
#define ESC_BOLD_RED_WHITE ESC_STD"31;49m"
#define ESC_BOLD ESC"[1;"
#define RED ESC_STD_RED
#define BLUE ESC"[34m"
#define BLUE_BOLD ESC_BOLD"34m"
#define BLUE_DIM ESC_DIM"34m"
#define RESET ESC"[0m"