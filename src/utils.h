#ifndef UTILS_H
#define UTILS_H

#define SIMD
#define SIMD_AVX2

#include <stdlib.h>
#include <x86intrin.h>

#define inspect(f, e) printf(#e"=" f "\n", e)
#define range(i, a, b) (size_t i = a; i < b; i++)

#ifdef SIMD
// Vector of 4 doubles
typedef union { 
    __m256d vec;
    double  arr[4];
} Vec4d;

typedef struct {
    size_t len;

    size_t vecs_l;
    Vec4d *vecs;
} Vec;
#else

typedef struct {
    size_t len;
    double *arr;
} Vec;
#endif


Vec vec_new(size_t len);
void vec_destroy(Vec *v);
void vec_set(Vec *v, double *arr);
void vec_set_i(Vec *v, size_t i, double val);
void vec_set1(Vec *v, double x);
void vec_set_rand(Vec *v);
void vec_set_vec(Vec *dst, Vec *src);

double *vec_as_arr(Vec *v);

#define VEC_OP_DECL(n) void vec_##n(Vec *a, Vec *b, Vec *dst);

VEC_OP_DECL(mul)
VEC_OP_DECL(div)
VEC_OP_DECL(add)
VEC_OP_DECL(sub)
void vec_nexp(Vec *x, Vec *dst);

#undef VEC_OP_DECL

double vec_fold(Vec *v);
void vec_to_arr(Vec *v, double *arr);
double vec_get_i(Vec *v, size_t i);

#endif
