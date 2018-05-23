#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils.h"
#include "network.h"
#define data_len 2

#define GRAPH_MODE

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

int main(int argc, char** argv) {
    srand(time(NULL));

    Network net = nw_new(data_len, data_len, 2, 1, 0.5);

    double input[data_len];
    double output[data_len];

    gen_arrays(data_len, input, output);

    InputLayer input_l = input_from(input, data_len);

    nw_initialize_nodes(&net);
    nw_load_input(&net, &input_l);

    double max_err = 0.0;
    size_t tt = 0;
    for range(c, 0, 10000) {
        gen_arrays(data_len, input, output);

#ifndef GRAPH_MODE
        print_arr("input:      ", data_len,  input, "\n");
        print_arr("target out: ", data_len, output, "\n");
#endif

        for range(t, 0, 600) {
            nw_forward_pass(&net);
            nw_backprop(&net, output);

#ifdef GRAPH_MODE
            if (t == 0) {
                max_err = nw_get_total_err(&net, output);
            printf("%lu\t"
                   "%lu\t"
                   "%lu\t"
                   "%lf\t"
                   "%lf\n",
                   tt,
                   c,
                   t,
                   nw_get_total_err(&net, output),
                   max_err
                   );
            }
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

    nw_destroy(&net);

    return 0;
}
