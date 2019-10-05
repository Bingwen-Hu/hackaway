#include <stdio.h>
#include "xmalloc.h"

// generic function
#define make_vector(v, n)  \
    ((v) = xmalloc((n) * sizeof *(v)))

#define free_vector(v) \
    do { free(v); v = NULL; } while (0)

#define make_matrix(a, m, n)                                    \
    do {                                                        \
        size_t make_matrix_loop_counter;                        \
        make_vector(a, (m) + 1);                                \
        for (make_matrix_loop_counter = 0;                      \
             make_matrix_loop_counter < (m);                    \
             make_matrix_loop_counter++) {                      \
            make_vector((a)[make_matrix_loop_counter], (n));    \
        }                                                       \
        (a)[m] = NULL;                                          \
    } while (0)

#define free_matrix(a)                                          \
    do {                                                        \
        if (a != NULL) {                                        \
            size_t make_matrix_loop_counter;                    \
            for (make_matrix_loop_counter = 0;                  \
                 (a)[make_matrix_loop_counter] != NULL;         \
                 make_matrix_loop_counter++) {                  \
                free_vector((a)[make_matrix_loop_counter]);     \
            }                                                   \
            free_vector(a);                                     \
            a = NULL;                                           \
        }                                                       \
    } while (0)

#define fprint_vector(stream, fmt, v, n)                                \
    do {                                                                \
        size_t fprint_vector_loop_counter;                              \
        for (fprint_vector_loop_counter = 0;                            \
             fprint_vector_loop_counter < (n);                          \
             fprint_vector_loop_counter++) {                            \
            fprintf(stream, fmt, (v)[fprint_vector_loop_counter]);      \
        }                                                               \
        fputc('\n', stream);                                            \
    } while (0)


#define print_vector(fmt, v, n) fprint_vector(stdout, fmt, v, n)

#define fprint_matrix(stream, fmt, a, m, n)                             \
    do {                                                                \
        size_t fprint_matrix_loop_counter;                              \
        for (fprint_matrix_loop_counter = 0;                            \
             fprint_matrix_loop_counter < (m);                          \
             fprint_matrix_loop_counter++) {                            \
            fprint_vector(stream, fmt,                                  \
                (a)[fprint_matrix_loop_counter], n);                    \
        }                                                               \
        fputc('\n', stream);                                            \
    } while (0)

#define print_matrix(fmt, a, m, n) fprint_matrix(stdout, fmt, a, m, n)