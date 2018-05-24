#define GRAPH_MODE
#define SIMD
#define SIMD_AVX2

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils.h"
#include "network.h"
#define data_len 2

void print_arr(char *pre, size_t len, double *arr, char *post) {
    if (pre != NULL)
        printf("%s", pre);
    printf("[ ");
    for range(i, 0, len) {
        printf("%lf ", arr[i]);
    }
    printf("]");
    if (post != NULL)
        printf("%s", post);
}

void gen_arrays(size_t len, double *dest_a, double *dest_b) {
    for range(i, 0, len) {
        double v = rand() / (double)RAND_MAX;
        dest_a[i] = v;
        dest_b[len-i-1] = v;
    }
}

void test_simd() {
    Vec v = vec_new(10);
    Vec x = vec_new(10);
    Vec o = vec_new(10);

    double arr[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    vec_set(&v, arr);
    vec_as_arr(&v)[5] = 80;

    print_arr("v: ", 10, arr, "\n");

    vec_set(&x, arr);
    vec_mul(&v, &x, &o);
    /* vec_nexp(&o, &o); */
    sigma(&o, &o);

    /* vec_to_arr(&o, arr); */
    double *out = vec_as_arr(&o);
    print_arr("o: ", 10, out, "\n");
    /* print_arr("o: ", 10, arr, "\n"); */

    vec_destroy(&v);
    vec_destroy(&x);
    vec_destroy(&o);

    /* Vec4d v = { .arr = { 1, 2, 3, 4 } }; */
    /* Vec4d x = { .arr = { 1, 2, 3, 4 } }; */
    /* Vec4d o = { 0 }; */

    /* print_arr("v: ", 4, v.arr, "\n"); */

    /* o.vec = _mm256_mul_pd(v.vec, x.vec); */
    /* /1* for range(i, 0, 4) { *1/ */
    /* /1*     o.arr[i] = v.arr[i] * x.arr[i]; *1/ */
    /* /1* } *1/ */

    /* print_arr("o: ", 4, o.arr, "\n"); */
}

int main(int argc, char** argv) {
    /* test_simd(); */
    /* return 0; */

    srand(time(NULL));

    Network net = nw_new(data_len, data_len, 2, 1, 0.5);

    /* double input[data_len]; */
    /* double output[data_len]; */
    Vec *input_vec = malloc(sizeof(Vec));
    *input_vec = vec_new(data_len);
    Vec output_vec = vec_new(data_len);
    double *input  = vec_as_arr(input_vec);
    double *output = vec_as_arr(&output_vec);
    /* vec_set(&input_vec, input); */
    /* vec_set(&output_vec, output); */

    gen_arrays(data_len, input, output);

    /* InputLayer input_l = input_from(input, data_len); */

    nw_initialize_nodes(&net);
    nw_load_input(&net, input_vec);

    double max_err = 0.0;
    size_t tt = 0;
    for range(c, 0, 1000) {
        gen_arrays(data_len, input, output);

#ifndef GRAPH_MODE
        print_arr("input:      ", data_len,  input, "\n");
        print_arr("target out: ", data_len, output, "\n");
#endif

        for range(t, 0, 700) {
            nw_forward_pass(&net);
            nw_backprop(&net, &output_vec);

#ifdef GRAPH_MODE
            if (t == 0) {
                max_err = nw_get_total_err(&net, &output_vec);
            }
            printf("%lu\t"
                   "%lu\t"
                   "%lu\t"
                   "%lf\t"
                   "%lf\n",
                   tt,
                   c,
                   t,
                   nw_get_total_err(&net, &output_vec),
                   max_err
                   );
            /* } */
#endif
            tt++;
        }

#ifndef GRAPH_MODE
        nw_forward_pass(&net);
        printf("output:     [ ");
        for range(i, 0, net.output->len) {
            printf("%lf ", net.output->nodes[i].out);
        }
        puts("]");

        puts("---");
#endif


    }

    vec_destroy(&output_vec);
    nw_destroy(&net);

    return 0;
}
