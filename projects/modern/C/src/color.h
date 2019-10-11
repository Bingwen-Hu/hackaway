#pragma once


#define ESC "\033"
#define STD ESC"[0;"
#define BOLD ESC"[1;"
#define DIM ESC"[2;"


#define RED_STD STD"31m"
#define RED_BOLD STD"31m"

#define ESC_STD_RED_WHITE ESC_STD"31;47m"
#define ESC_BOLD_RED_WHITE ESC_STD"31;49m"
#define ESC_BOLD ESC"[1;"
#define RED ESC_STD_RED
#define BLUE ESC"[34m"
#define BLUE_BOLD ESC_BOLD"34m"
#define BLUE_DIM ESC_DIM"34m"
#define RESET ESC"[0m"