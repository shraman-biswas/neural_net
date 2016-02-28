#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

typedef struct __nn_matrix_t {
	int rows, cols;
	double *data;
} nn_matrix_t;

typedef struct __neural_net_t {
	int num;
	const int *layers;
	double step, range;
	gsl_rng *rng;
	gsl_matrix **wts, **dwt, **dlt, **act;
} neural_net_t;

neural_net_t *nn_create(
	const int *layers,
	const int num,
	const double step,
	const double range);
void nn_destroy(neural_net_t *nn);
void nn_train(
	neural_net_t *nn,
	const gsl_matrix *in,
	const gsl_matrix *tgt,
	const int epochs);
void nn_predict(
	neural_net_t *nn,
	const gsl_matrix *in,
	const gsl_matrix *res);
void matrix_print(const gsl_matrix *m);
gsl_matrix *arr_to_gslmat(
	const double *arr,
	const int rows,
	const int cols);

#endif
