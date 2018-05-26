#include <stdlib.h>
#include <stdio.h>
#include "network.h"
#include "utils.h"

#include <math.h>

/* inline double sigma(double x) { */
/*     return 1.0 / (1 + exp(-x)); */
/* } */

void sigma(Vec *x, Vec *y, Vec *tmp) {
    /* Vec tmp = vec_new(y->len); */

    vec_set1(tmp, 1.0);

    vec_nexp(x, y);
    vec_add(tmp, y, y);
    vec_div(tmp, y, y);

    /* vec_destroy(tmp); */
}

/* Node node_new(size_t weights_l) { */
/*     Node n = { 0 }; */

/*     n.weights = malloc(sizeof(double)*weights_l); */
/*     n.weights_l = weights_l; */

/*     return n; */
/* } */

/* void node_destroy(Node *n) { */
/*     if (n->weights != NULL) { */
/*         free(n->weights); */
/*         n->weights   = NULL; */
/*         n->weights_l = 0; */
/*     } */
/* } */

/* void node_initialize(Node *n) { */
/*     if (n->weights != NULL) { */
/*         for range(i, 0, n->weights_l) { */
/*             n->weights[i] = rand() / (double)RAND_MAX; */
/*             /1* printf("random weight: %lf (%lu)\n", n->weights[i], i); *1/ */
/*         } */
/*     } */
/* } */

/* InputLayer input_new(size_t len) { */
/*     InputLayer in = { 0 }; */

/*     in.len = len; */
/*     in.nodes = malloc(sizeof(Node)*len); */

/*     return in; */
/* } */

/* InputLayer input_from(double *arr, size_t len) { */
/*     InputLayer in = { */
/*         .len = len, */
/*         .nodes = arr, */
/*     }; */

/*     return in; */
/* } */


/* void input_destroy(InputLayer *in) { */
/*     if (in->values != NULL) { */
/*         free(in->values); */
/*         in->values = NULL; */
/*         in->len   = 0; */
/*     } */
/* } */

Layer layer_new(size_t len, size_t next_len) {
    Layer l = { 0 };

    l.len = len;
    l.out   = malloc(sizeof(Vec));
    *l.out  = vec_new(len);
    l.net   = vec_new(len);
    l.bias  = vec_new(len);
    l.error = vec_new(len);

    l.tmp1  = vec_new(len);
    l.tmp2  = vec_new(len);

    l.weights_l = next_len;
    l.weights = malloc(sizeof(Vec)*next_len);
    for range(i, 0, next_len) {
        l.weights[i] = vec_new(len);
    }

    /* l.nodes = malloc(sizeof(Node)*len); */

    /* for range(i, 0, len) { */
    /*     l.nodes[i] = node_new(prev_len); */
    /* } */

    return l;
}

void layer_destroy(Layer *l) {
    if (l->len != 0) {
        vec_destroy(l->out);
        free(l->out);
        vec_destroy(&l->net);
        vec_destroy(&l->bias);
        vec_destroy(&l->error);

        vec_destroy(&l->tmp1);
        vec_destroy(&l->tmp2);

        for range(i, 0, l->weights_l) {
            vec_destroy(l->weights+i);
        }
        free(l->weights);

        l->len = 0;
        l->weights_l = 0;
    }
    /* for range(i, 0, l->len) { */
    /*     node_destroy(l->nodes + i); */
    /* } */

    /* if (l->nodes != NULL) { */
    /*     free(l->nodes); */
    /*     l->nodes = NULL; */
    /*     l->len   = 0; */
    /* } */

}

void layer_randomize_weights(Layer *l) {
    for range(n, 0, l->weights_l)
        vec_set_rand(l->weights+n);

}

void layer_randomize_biases(Layer *l) {
    vec_set_rand(&l->bias);
    /* for range(i, 0, l->len) { */
    /*     /1* node_initialize(l->nodes + i); *1/ */
    /*     for range(n, 0, l->weights_l_l) */
    /*         l->weights[i][n] = rand() / (double)RAND_MAX; */
    /*     l->bias[i] = rand() / (double)RAND_MAX; */
    /* } */
}

Network nw_new(size_t ins, size_t outs, size_t hid, size_t h_layers, double learning_rate) {
    Network nw = { 0 };

    // Input Layer 
    nw.input = malloc(sizeof(InputLayer));
    *nw.input = layer_new(ins, hid);

    // Output Layer`
    nw.output = malloc(sizeof(OutputLayer));
    *nw.output = layer_new(outs, 0);

    // Hidden Layers
    nw.hidden = malloc(sizeof(HiddenLayer)*h_layers);
    nw.hidden_l = h_layers;
    for range(i, 0, h_layers) {
        nw.hidden[i] = layer_new(hid, i == h_layers-1 ? outs : hid);
    }

    // All Layers
    size_t layers_l = h_layers + 2;
    nw.layers = malloc(sizeof(Layer*)*layers_l);
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
        layer_destroy(nw->input);
        free(nw->input);
        nw->input = NULL;
    }
}

void nw_initialize_nodes(Network *nw) {
    for range(i, 0, nw->layers_l-1) {
        /* printf("initializing layer: %lu/%lu\n", i+1, nw->layers_l); */

        /* layer_initialize_nodes(nw->layers[i]); */
        layer_randomize_weights(nw->layers[i]);
        layer_randomize_biases(nw->layers[i+1]);
    }
}

inline void nw_load_input(Network *nw, Vec *in) {
    vec_destroy(nw->input->out);
    nw->input->out = in;
}

void nw_forward_pass(Network *nw) {
    for range(i, 1, nw->layers_l) {
        Layer *prev = nw->layers[i-1];
        Layer *curr = nw->layers[i];


        /* Vec tmp = vec_new(curr->net.len); */
        /* Vec *tmp = &curr->tmp1; */
        Vec *tmp = &prev->tmp1;
        for range(t, 0, curr->len) {
            vec_mul(prev->out, &prev->weights[t], tmp);
            vec_set_i(&curr->net, t, vec_fold(tmp));
        }

        vec_add(&curr->net, &curr->bias, &curr->net);

        sigma(&curr->net, curr->out, &curr->tmp2);
        /* inspect("%lf", vec_as_arr(&curr->net)[0]); */

        /* vec_destroy(&tmp); */

        /* cur->net */
        /* if (i == 0) { */ 
        /*     InputLayer *layer = *(InputLayer **)*(nw->layers+i); */
        /*     for range(n, 0, next->len) { */

        /*         double sum = next->bias[n]; */
        /*         for range(pn, 0, layer->len) { */
        /*             sum += layer->values[pn] * next->nodes[n].weights[pn]; */
        /*         } */

        /*         next->nodes[n].net = sum; */
        /*         next->nodes[n].out = sigma(sum); */
        /*     } */
        /* } else { */
        /*     Layer *layer = (Layer *)*(nw->layers+i); */
        /*     for range(n, 0, next->len) { */

        /*         double sum = next->nodes[n].bias; */
        /*         for range(pn, 0, layer->len) { */
        /*             sum += layer->nodes[pn].out * next->nodes[n].weights[pn]; */
        /*         } */

        /*         next->nodes[n].net = sum; */
        /*         next->nodes[n].out = sigma(sum); */
        /*     } */
        /* } */
    }
}

void nw_backprop(Network *nw, Vec *target_output) {
    for (size_t i = nw->layers_l-1; i>=1; i--) {
        Layer *layer = (Layer *)*(nw->layers + i);
        Layer *prev  = (Layer *)*(nw->layers + i -1);

        /* if (i == nw->layers_l-1) { */
        if (layer->weights_l == 0) {
            // layer->error =
            //      layer->out * (1 - layer->out)
            //                 * (layer->out - target_output);

            // (layer->out - target_output)
            vec_sub(layer->out, target_output, &layer->error);

            // * (1 - layer->out)
            /* Vec tmp1 = vec_new(layer->len); */
            Vec *tmp1 = &layer->tmp1;
            vec_set1(tmp1, 1);
            vec_sub(tmp1, layer->out, tmp1);
            vec_mul(tmp1, &layer->error, &layer->error);

            // * layer->out
            vec_mul(layer->out, &layer->error, &layer->error);

            /* vec_destroy(&tmp1); */
        } else { 
            Layer *next = (Layer *)*(nw->layers + i +1);
            // out * (1-out) * SUM_a ( next_err * W_here_previous )
            // layer->error = 
            //      layer->out * (1 - layer->out)
            //                 * fold( next->err[a] * layer->weights[next_index])
            
            // (layer->out - target_output)
            vec_sub(layer->out, target_output, &layer->error);

            // * (1 - layer->out)
            /* Vec tmp1 = vec_new(layer->len); */
            Vec *tmp1 = &layer->tmp1;
            vec_set1(tmp1, 1);
            vec_sub(tmp1, layer->out, tmp1);
            vec_mul(tmp1, &layer->error, &layer->error);

            /* Vec tmp2 = vec_new(layer->len); */
            /* Vec tmp3 = vec_new(layer->len); */
            Vec *tmp2 = &layer->tmp2;
            vec_set1(tmp2, 0);
            for range(a, 0, next->len) {
                vec_set1(tmp1, vec_get_i(&next->error, a));

                /* for range(ii, 0, layer->len) { */
                /*     vec_set_i(&tmp2, ii, vec_get_i(&layer->weights[ii], a)); */
                /* } */

                /* vec_mul(&tmp1, &tmp2, &tmp1); */
                vec_mul(tmp1, &layer->weights[a], tmp1);

                vec_add(tmp1, tmp2, tmp2);

            }
            vec_mul(&layer->error, tmp2, &layer->error);

            /* vec_destroy(&tmp1); */
            /* vec_destroy(&tmp2); */
            /* vec_destroy(&tmp3); */
        }

        // layer->weights[t] = layer->weights[t] - layer->error * prev->out * learning_rate
        /* Vec tmp1 = vec_new(layer->len); */
        /* Vec tmp2 = vec_new(layer->len); */
        Vec *tmp1 = &layer->tmp1;
        Vec *tmp2 = &layer->tmp2;
        vec_set1(tmp2, nw->learning_rate);
        for range(t, 0, layer->len) {
            vec_set1(tmp1, vec_get_i(&layer->error, t));
            vec_mul(prev->out,     tmp1, tmp1);
            vec_mul(tmp1, tmp2, tmp1);
            /* vec_mul(&layer->error, &tmp1, &tmp1); */
            /* print_arr("tmp2  = ", layer->len, vec_as_arr(tmp2), "\n"); */


            vec_sub(&prev->weights[t], tmp1, &prev->weights[t]);
        }
        /* puts("--"); */

        // layer->bias = layer->bias - layer->error * learning_rate
        vec_mul(&layer->error, tmp2, tmp1);
        vec_sub(&layer->bias, tmp1, &layer->bias);
        
        /* vec_destroy(&tmp1); */
        /* vec_destroy(&tmp2); */

    }

    /*
    for (int i = nw->layers_l-1; i >= 1; i--) {
        Layer *layer = (Layer *)*(nw->layers + i);
        for range(n, 0, layer->len) {
            Node *node = (Node *)(layer->nodes + n);

            // === Backprop Errors ===
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
                    //     next->nodes[nn].error * node->weights[nn]
                }

                node->error =
                    node->out
                    * (1.0 - node->out)
                    * sum;
                // out * (1-out) * SUM_a ( next_err * W_here_previous )
            }

            // === Update Weights ===
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
    */
}

double nw_get_total_err(Network *nw, Vec *target_output) {
    /* Vec tmp = vec_new(nw->output->len); */
    Vec *tmp = &nw->output->tmp1;

    vec_sub(target_output, nw->output->out, tmp);
    vec_mul(tmp, tmp, tmp);
    double r = vec_fold(tmp);

    /* vec_destroy(tmp); */
    return r / 2.0;

    /* double sum = 0.0; */
    /* for range(i, 0, nw->output->len) { */
    /*     sum += pow(nw->output->nodes[i].out - target_output[i], 2); */
    /* } */

    /* return sum / 2.0; */
}
