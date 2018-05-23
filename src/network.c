#include <stdlib.h>
#include <stdio.h>
#include "network.h"
#include "utils.h"

#include <math.h>

inline double sigma(double x) {
    return 1.0 / (1 + exp(-x));
}

Node node_new(size_t weights_l) {
    Node n = { 0 };

    n.weights = malloc(sizeof(double)*weights_l);
    n.weights_l = weights_l;

    return n;
}

void node_destroy(Node *n) {
    if (n->weights != NULL) {
        free(n->weights);
        n->weights   = NULL;
        n->weights_l = 0;
    }
}

void node_initialize(Node *n) {
    if (n->weights != NULL) {
        for range(i, 0, n->weights_l) {
            n->weights[i] = rand() / (double)RAND_MAX;
            /* printf("random weight: %lf (%lu)\n", n->weights[i], i); */
        }
    }
}

InputLayer input_new(size_t len) {
    InputLayer in = { 0 };

    in.len = len;
    in.values = malloc(sizeof(double)*len);

    return in;
}

InputLayer input_from(double *arr, size_t len) {
    InputLayer in = {
        .len = len,
        .values = arr,
    };

    return in;
}

void input_destroy(InputLayer *in) {
    if (in->values != NULL) {
        free(in->values);
        in->values = NULL;
        in->len   = 0;
    }
}

Layer layer_new(size_t len, size_t prev_len) {
    Layer l = { 0 };

    l.len = len;
    l.nodes = malloc(sizeof(Node)*len);

    for range(i, 0, len) {
        l.nodes[i] = node_new(prev_len);
    }

    return l;
}

void layer_destroy(Layer *l) {
    for range(i, 0, l->len) {
        node_destroy(l->nodes + i);
    }

    if (l->nodes != NULL) {
        free(l->nodes);
        l->nodes = NULL;
        l->len   = 0;
    }

}

void layer_initialize_nodes(Layer *l) { 
    for range(i, 0, l->len) {
        node_initialize(l->nodes + i);
    }
}

Network nw_new(size_t ins, size_t outs, size_t hid, size_t h_layers, double learning_rate) {
    Network nw = { 0 };

    // Input Layer placeholder
    nw.input = malloc(sizeof(Layer *));
    *nw.input = NULL;

    // Output Layer`
    nw.output = malloc(sizeof(Layer));
    *nw.output = layer_new(outs, hid);

    // Hidden Layers
    nw.hidden = malloc(sizeof(HiddenLayer)*h_layers);
    nw.hidden_l = h_layers;
    for range(i, 0, h_layers) {
        nw.hidden[i] = layer_new(hid, i == 0 ? ins : hid);
    }

    // All Layers
    size_t layers_l = h_layers + 2;
    nw.layers = malloc(sizeof(void*)*layers_l);
    nw.layers_l = layers_l;
    for range(i, 0, layers_l) {
        if (i == 0) {
            nw.layers[i] = nw.input;
        } else if (i == layers_l-1) {
            nw.layers[i] = nw.output;
        } else {
            nw.layers[i] = nw.hidden+i-1;
        }
    }


    nw.learning_rate = learning_rate;

    return nw;
}

void nw_destroy(Network *nw) {
    // Layers array
    if (nw->layers != NULL) {
        free(nw->layers);
        nw->layers = NULL;
        nw->layers_l = 0;
    }

    // Hidden layers
    if (nw->hidden != NULL) {
        for range(i, 0, nw->hidden_l) {
            layer_destroy(nw->hidden + i);
        }
        free(nw->hidden);
        nw->hidden = NULL;
        nw->hidden_l = 0;
    }

    // Output layer
    if (nw->output != NULL) {
        layer_destroy(nw->output);
        free(nw->output);
        nw->output = NULL;
    }

    // Input layer placeholder
    if (nw->input != NULL) {
        free(nw->input);
        nw->input = NULL;
    }
}

void nw_initialize_nodes(Network *nw) {
    for range(i, 1, nw->layers_l) {
        /* printf("initializing layer: %lu/%lu\n", i+1, nw->layers_l); */

        layer_initialize_nodes(nw->layers[i]);
    }
}

inline void nw_load_input(Network *nw, InputLayer *in) {
    *nw->input = in;
}

void nw_forward_pass(Network *nw) {
    for range(i, 0, nw->layers_l-1) {
        Layer *next = (Layer *)*(nw->layers+i+1);
        if (i == 0) { 
            InputLayer *layer = *(InputLayer **)*(nw->layers+i);
            for range(n, 0, next->len) {

                double sum = next->nodes[n].bias;
                for range(pn, 0, layer->len) {
                    sum += layer->values[pn] * next->nodes[n].weights[pn];
                }

                next->nodes[n].net = sum;
                next->nodes[n].out = sigma(sum);
            }
        } else {
            Layer *layer = (Layer *)*(nw->layers+i);
            for range(n, 0, next->len) {

                double sum = next->nodes[n].bias;
                for range(pn, 0, layer->len) {
                    sum += layer->nodes[pn].out * next->nodes[n].weights[pn];
                }

                next->nodes[n].net = sum;
                next->nodes[n].out = sigma(sum);
            }
        }
    }
}

void nw_backprop(Network *nw, double *target_output) {
    for (int i = nw->layers_l-1; i >= 1; i--) {
        Layer *layer = (Layer *)*(nw->layers + i);
        for range(n, 0, layer->len) {
            Node *node = (Node *)(layer->nodes + n);

            if (i == nw->layers_l-1) {
                node->error =
                    node->out
                    * (1.0 - node->out) 
                    * (node->out - target_output[n]);
            } else {
                double sum = 0.0;
                Layer *next = (Layer *)*(nw->layers + i +1);
                for range(nn, 0, next->len) {
                    sum += next->nodes[nn].error * next->nodes[nn].weights[n];
                }

                node->error =
                    node->out
                    * (1.0 - node->out)
                    * sum;
                // out * (1-out) * SUM_a ( next_err * W_here_previous )
            }

            if (i == 1) {
                InputLayer *previous = *(InputLayer **)*(nw->layers + i -1);
                for range(pn, 0, previous->len) {
                    node->weights[pn] += - node->error 
                        * previous->values[pn] 
                        * nw->learning_rate;
                }
            } else {
                Layer *previous = (Layer *)*(nw->layers + i -1);
                for range(pn, 0, previous->len) {
                    node->weights[pn] += - node->error 
                        * previous->nodes[pn].out
                        * nw->learning_rate;
                }
            }
        }
    }
}

double nw_get_total_err(Network *nw, double *target_output) { 
    double sum = 0.0;
    for range(i, 0, nw->output->len) {
        sum += pow(nw->output->nodes[i].out - target_output[i], 2);
    }

    return sum / 2.0;
}
