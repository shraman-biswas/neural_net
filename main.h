#ifndef MAIN_H
#define MAIN_H

#include <stdio.h>
#include <stdlib.h>

#include <gsl/gsl_matrix.h>

#include "neural_net.h"

#define SIZE(x) (sizeof(x) / sizeof(*(x)))

const int layers[] = {2, 2, 1};			/* neural network layers */
const double train_arr[] = {			/* training inputs array */
	0,	0,
	0, 	1,
	1, 	0,
	1,	1,
};
const double target_arr[] = {			/* training targets array */
	0,
	1,
	1,
	0,
};

static void disp_result(
	const gsl_matrix *test,
	const gsl_matrix *result,
	const int set);
static void select_test(
	const gsl_matrix *train,
	gsl_matrix *test,
	const int set);

#endif
