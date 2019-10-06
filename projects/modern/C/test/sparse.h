// CCS representation
#pragma once


void sparse_pack(double** a, int m, int n,
        int* Ap, int* Ai, double* Ax);

void sparse_unpack(double** a, int m, int n, 
        int* Ap, int* Ai, double* Ax);
