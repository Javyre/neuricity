#ifndef NETWORK_H
#define NETWORK_H

#include <stdlib.h>

typedef struct {
    double out;
    double net;
    double bias;
    double error;

    size_t weights_l;
    double *weights;
} Node;

typedef struct {
    size_t len;
    double *values;
} InputLayer;

typedef struct {
    size_t len;
    Node   *nodes;
} Layer;

typedef Layer OutputLayer;
typedef Layer HiddenLayer;

typedef struct {
    size_t hidden_l;
    size_t layers_l;
    double learning_rate;

    InputLayer   **input;
    OutputLayer  *output;
    HiddenLayer  *hidden;

    void **layers;
    /*
     * void **layers = { **input, *hidden + i ..., *output }
     *             0 = ***InputLayer
     *             i = ** HiddenLayer
     *            -1 = ** OutputLayer
     */
} Network;

double sigma(double);

Node node_new(size_t weights_l);
void node_destroy(Node *n);
void node_initialize(Node *n);

InputLayer input_new(size_t len);
InputLayer input_from(double *arr, size_t len);
void input_destroy(InputLayer *in);

Layer layer_new(size_t len, size_t prev_len);
void layer_destroy(Layer *l);
void layer_initialize_nodes(Layer *l);

Network nw_new(size_t ins, size_t outs, size_t hid, size_t h_layers, double learning_rate);
void nw_destroy(Network *nw);

void nw_initialize_nodes(Network *nw);
void nw_load_input(Network *nw, InputLayer *in);
void nw_forward_pass(Network *);
void nw_backprop(Network *, double *);
double nw_get_total_err(Network *, double *);

#endif
