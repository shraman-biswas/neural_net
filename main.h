#ifndef MAIN_H
#define MAIN_H

#include <stdio.h>
#include <stdlib.h>

#include <gsl/gsl_matrix.h>

#include "neural_net.h"

#define EPOCHS 10000
#define SIZE(x) (sizeof(x) / sizeof(*(x)))

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

static void disp_res(
	const gsl_matrix *in,
	const gsl_matrix *res,
	const int set);
static void set_x(
	const gsl_matrix *in,
	const gsl_matrix *x,
	const int set);

#endif
