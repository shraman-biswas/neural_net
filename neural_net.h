#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

/* neural network structure */
typedef struct __neural_net_t {
	int num;
	const int *layers;
	double step, range;
	gsl_rng *rng;
	gsl_matrix **wts, **dwt, **dlt, **act;
} neural_net_t;

neural_net_t *nn_create(
	const int *const layers,
	const int num,
	const double step,
	const double range);
void nn_destroy(neural_net_t *const nn);
void nn_train(
	neural_net_t *const nn,
	const gsl_matrix *const train,
	const gsl_matrix *const target,
	const int epochs);
void nn_predict(
	neural_net_t *const nn,
	const gsl_matrix *const x,
	gsl_matrix *const result);
void disp_matrix(const gsl_matrix *m);
gsl_matrix *arr_to_gslmat(
	const double *arr,
	const int rows,
	const int cols);

#endif
