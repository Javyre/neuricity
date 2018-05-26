#define GRAPH_MODE
#define SIMD
#define SIMD_AVX2

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils.h"
#include "network.h"
#define data_len 2


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

    Vec tmp = vec_new(10);
    sigma(&o, &o, &tmp);
    vec_destroy(&tmp);

    /* vec_to_arr(&o, arr); */
    double *out = vec_as_arr(&o);
    print_arr("o: ", 10, out, "\n");
    /* print_arr("o: ", 10, arr, "\n"); */

    vec_destroy(&v);
    vec_destroy(&x);
    vec_destroy(&o);
}

void test_simple() {
    srand(time(NULL));

    Network net = nw_new(data_len, data_len, 2, 1, 0.5);

    Vec *input_vec = malloc(sizeof(Vec));
    *input_vec = vec_new(data_len);
    Vec output_vec = vec_new(data_len);
    double *input  = vec_as_arr(input_vec);
    double *output = vec_as_arr(&output_vec);

    nw_load_input(&net, input_vec);

    input[0] = 0.05;
    input[1] = 0.10;
    output[0] = 0.01;
    output[1] = 0.99;

    vec_as_arr(&net.hidden->bias)[0] = 0.35;
    vec_as_arr(&net.hidden->bias)[1] = 0.35;

    vec_as_arr(&net.output->bias)[0] = 0.60;
    vec_as_arr(&net.output->bias)[1] = 0.60;

    vec_as_arr(&net.input->weights[0])[0] = 0.15;
    vec_as_arr(&net.input->weights[0])[1] = 0.20;
    vec_as_arr(&net.input->weights[1])[0] = 0.25;
    vec_as_arr(&net.input->weights[1])[1] = 0.30;

    vec_as_arr(&net.hidden->weights[0])[0] = 0.40;
    vec_as_arr(&net.hidden->weights[0])[1] = 0.45;
    vec_as_arr(&net.hidden->weights[1])[0] = 0.50;
    vec_as_arr(&net.hidden->weights[1])[1] = 0.55;

    nw_forward_pass(&net);
    print_arr("i:  ", 2, input, "\n");
    print_arr("to: ", 2, output, "\n");
    print_arr("o:  ", 2, vec_as_arr(net.output->out), "\n");

    nw_backprop(&net, &output_vec);
    print_arr("wi: ", 2, vec_as_arr(&net.input->weights[0]), "\n");
    print_arr("wi: ", 2, vec_as_arr(&net.input->weights[1]), "\n");
    print_arr("wh: ", 2, vec_as_arr(&net.hidden->weights[0]), "\n");
    print_arr("wh: ", 2, vec_as_arr(&net.hidden->weights[1]), "\n");

    vec_destroy(&output_vec);
    nw_destroy(&net);
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


    /* double max_err = 1.0; */
    size_t tt = 0;
    for range(c, 0, 1000) {
        gen_arrays(data_len, input, output);

#ifndef GRAPH_MODE
        print_arr("input:      ", data_len,  input, "\n");
        print_arr("target out: ", data_len, output, "\n");
#endif

        for range(t, 0, 100) {
            nw_forward_pass(&net);


            nw_backprop(&net, &output_vec);

#ifdef GRAPH_MODE
            /* if (t == 0) { */
            /*     max_err = nw_get_total_err(&net, &output_vec); */
            /* } */

            /* if (max_err < 0.005) { */
            /*     tt += 400-1; */
            /*     break; */
            /* } */

            printf("%lu\t"
                   "%lu\t"
                   "%lu\t"
                   /* "%lf\t" */
                   "%lf\n",
                   tt,
                   c,
                   t,
                   nw_get_total_err(&net, &output_vec)
                   /* max_err */
                   );
            /* } */
#endif
            tt++;
        }
        puts("");

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
