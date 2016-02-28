#ifndef MAIN_H
#define MAIN_H

#include <stdio.h>
#include <stdlib.h>

#include "neural_net.h"

#define EPOCHS 10000

const int layers[] = {2, 2, 1};
const double in_arr[] = {
	0,	0,
	0, 	1,
	1, 	0,
	1,	1
};
const double tgt_arr[] = {
	0,
	1,
	1,
	0
};

nn_matrix_t in = {4, 2, (double *)in_arr};
nn_matrix_t tgt = {4, 1, (double *)tgt_arr};

static void disp_res(const nn_matrix_t *res, const int set);
static void set_x(const nn_matrix_t *x, const int set);

#endif
