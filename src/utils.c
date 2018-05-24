#include "utils.h"
#include <x86intrin.h>
#include <string.h>
#include <math.h>

#ifdef SIMD

Vec vec_new(size_t len) {
    size_t count = (len / 4) + (len % 4 == 0 ? 0 : 1);
    Vec4d *sub_vecs = aligned_alloc(32, sizeof(Vec4d)*count);
    sub_vecs[count-1].vec = _mm256_set1_pd(0);

    Vec v = { len, count, sub_vecs };
    return v;
}

void vec_destroy(Vec *v) {
    v->len = v->vecs_l = 0;
    free(v->vecs);
}

void vec_set(Vec *v, double *arr) {
    for range(i, 0, v->len) {
        /* v->vecs[i/4].arr[i%4] = arr[i]; */
        vec_as_arr(v)[i] = arr[i];
    }
}

void vec_set_i(Vec *v, size_t i, double val) {
    /* v->vecs[i/4].arr[i%4] = val; */
    vec_as_arr(v)[i] = val;
}

void vec_set1(Vec *v, double x) {
    for range(s, 0, v->vecs_l) {
        v->vecs[s].vec = _mm256_set1_pd(x);
    }
}

// not vectorized yet (impossible with AVX2)
void vec_set_rand(Vec *v) {
    for range(i, 0, v->len) {
        /* v->vecs[i/4].arr[i%4] = rand() / RAND_MAX; */
        vec_as_arr(v)[i] = rand() / RAND_MAX;
    }
}

void vec_set_vec(Vec *dst, Vec *src) {
    memcpy(dst->vecs, src->vecs, sizeof(Vec4d)*dst->vecs_l);
}

double *vec_as_arr(Vec *v) {
    return (double *)v->vecs;
}

#define VEC_OP(n)                                                               \
    void vec_##n(Vec *a, Vec *b, Vec *dst) {                                    \
        for range(i, 0, dst->vecs_l) {                                          \
            dst->vecs[i].vec = _mm256_##n##_pd(a->vecs[i].vec, b->vecs[i].vec); \
        }                                                                       \
    }

VEC_OP(mul)
VEC_OP(div)
VEC_OP(add)
VEC_OP(sub)

void vec_nexp(Vec *x, Vec *dst) {
    for range(i, 0, dst->len) {
        /* dst->vecs[i/4].arr[i%4] = exp(-x->vecs[i/4].arr[i%4]); */
        vec_as_arr(dst)[i] = exp(-vec_as_arr(x)[i]);
    }
}

double vec_fold(Vec *v) {
    double r = 0.0;
    for range(i, 0, v->len) { 
        /* r += v->vecs[i/4].arr[i%4]; */
        r += vec_as_arr(v)[i];
    }
    return r;
}

void vec_to_arr(Vec *v, double *arr) {
    for range(i, 0, v->len) {
        /* arr[i] = v->vecs[i/4].arr[i%4]; */
        arr[i] = vec_as_arr(v)[i];
    }
}

inline double vec_get_i(Vec *v, size_t i) {
    /* return v->vecs[i/4].arr[i%4]; */
    return vec_as_arr(v)[i];
}

#else

Vec vec_new(size_t len) {
    double *arr = aligned_alloc(32, sizeof(double)*len);

    Vec v = { len, arr };
    return v;
}

void vec_destroy(Vec *v) {
    v->len = 0;
    free(v->arr);
}

void vec_set(Vec *v, double *arr) {
    memcpy(v->arr, arr, sizeof(double)*v->len);
}

void vec_set_i(Vec *v, size_t i, double val) {
    v->arr[i] = val;
}

void vec_set1(Vec *v, double x) {
    for range(i, 0, v->len)
        v->arr[i] = x;
}

void vec_set_rand(Vec *v) {
    for range(i, 0, v->len) 
        v->arr[i] = rand() / RAND_MAX;
}

void vec_set_vec(Vec *dst, Vec *src) {
    memcpy(dst->arr, src->arr, sizeof(double)*dst->len);
}

inline double *vec_as_arr(Vec *v) {
    return v->arr;
}

#define VEC_OP(n, op)                             \
    void vec_##n(Vec *a, Vec *b, Vec *dst) {      \
        for range(i, 0, dst->len) {               \
            dst->arr[i] = a->arr[i] op b->arr[i]; \
        }                                         \
    }

VEC_OP(mul, *)
VEC_OP(div, /)
VEC_OP(add, +)
VEC_OP(sub, -)

void vec_nexp(Vec *x, Vec *dst) {
    for range(i, 0, x->len) {
        dst->arr[i] = exp(-x->arr[i]);
    }
}


double vec_fold(Vec *v) {
    double r = 0.0;
    for range(i, 0, v->len)
        r += v->arr[i];
    return r;
}

void vec_to_arr(Vec *v, double *arr) {
    memcpy(arr, v->arr, sizeof(double)*v->len);
}

double vec_get_i(Vec *v, size_t i) {
    return v->arr[i];
}

#endif
