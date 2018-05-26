#ifndef NETWORK_H
#define NETWORK_H

#include <stdlib.h>

#include <x86intrin.h>
#include "utils.h"

/* typedef struct { */
/*     double out; */
/*     double net; */
/*     double bias; */
/*     double error; */

/*     size_t   weights_l; */
/*     double  *weights; */
/* } Node; */

typedef struct {
    size_t len;

    Vec *out;
    Vec net;
    Vec bias;
    Vec error;
    /* double *out; */
    /* double *net; */
    /* double *bias; */
    /* double *error; */

    Vec tmp1;
    Vec tmp2;

    size_t weights_l;
    Vec *weights; // weights from here to next layer nodes
    // weights[next_node][this layer]

    /* Node   *nodes; */
} Layer;

typedef Layer InputLayer;
typedef Layer OutputLayer;
typedef Layer HiddenLayer;

typedef struct {
    size_t hidden_l;
    size_t layers_l;
    double learning_rate;

    InputLayer   *input;
    OutputLayer  *output;
    HiddenLayer  *hidden;


    Layer **layers;
    /*
     * void **layers = { **input, *hidden + i ..., *output }
     *             0 = ***InputLayer
     *             i = ** HiddenLayer
     *            -1 = ** OutputLayer
     */
} Network;

void sigma(Vec *x, Vec *y, Vec *tmp); // output is y

/* Node node_new(size_t weights_l); */
/* void node_destroy(Node *n); */
/* void node_initialize(Node *n); */

/* InputLayer input_new(size_t len); */
/* InputLayer input_from(double *arr, size_t len); */
/* void input_destroy(InputLayer *in); */

Layer layer_new(size_t len, size_t prev_len);
void layer_destroy(Layer *l);
void layer_randomize_weights(Layer *l);
void layer_randomize_biases(Layer *l);

Network nw_new(size_t ins, size_t outs, size_t hid, size_t h_layers, double learning_rate);
void nw_destroy(Network *nw);

void nw_initialize_nodes(Network *nw);
void nw_load_input(Network *nw, Vec *in);
void nw_forward_pass(Network *);
void nw_backprop(Network *, Vec *);
double nw_get_total_err(Network *, Vec *);

#endif
