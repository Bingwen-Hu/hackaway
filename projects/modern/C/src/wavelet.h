#pragma once

#define WT_FWD 0
#define WT_REV 1

void haar_transform_vector(double* v, int n, int dir);
void haar_transform_matrix(double** a, int m, int n, int dir);

