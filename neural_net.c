#include "neural_net.h"

static void matrix_print(const gsl_matrix *m)
{
	int i, j;
	for (i=0; i< m->size1; ++i) {
		for (j=0; j < m->size2-1; ++j) {
			printf("%f, ", gsl_matrix_get(m, i, j));
		}
		printf("%f\n", gsl_matrix_get(m, i, j));
	}
	printf("\n");
}

static void activ(gsl_matrix *m)
{
	int i;
	double tmp;
	for (i=0; i < m->size2; ++i) {
		tmp = tanh(gsl_matrix_get(m, 0, i));
		gsl_matrix_set(m, 0, i, tmp);
	}
}

static void activ_der(gsl_matrix *m)
{
	int i;
	double tmp;
	for (i=0; i < m->size2; ++i) {
		tmp = tanh(gsl_matrix_get(m, 0, i));
		gsl_matrix_set(m, 0, i, (1 - tmp * tmp));
	}
}

static void init_rand(neural_net_t *nn)
{
	gsl_rng_env_setup();
	nn->rng = gsl_rng_alloc(gsl_rng_default);
}

static void wts_rand(neural_net_t *nn, const int idx)
{
	int i, j;
	double tmp;
	gsl_matrix *wts = nn->wts[idx];
	for (i=0; i < wts->size1; ++i) {
		for (j=0; j < wts->size2; ++j) {
			tmp = 2 * gsl_rng_uniform(nn->rng) - 1;
			gsl_matrix_set(wts, i, j, tmp * nn->range);
		}
	}
}

static void matrix_append(gsl_matrix *m, const gsl_matrix *in)
{
	int i, j, tmp;
	for (i=0; i < in->size1; ++i) {
		for (j=0; j < in->size2+1; ++j) {
			tmp = (j < in->size2) ? in->data[i * in->size2 + j] : 1;
			gsl_matrix_set(m, i, j, tmp);
		}
	}
}

static void fwd_prop(neural_net_t *nn)
{
	int i;
	for (i=1; i < nn->num; ++i) {
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, \
		1, nn->act[i-1], nn->wts[i-1], 0, nn->act[i]);
		activ(nn->act[i]);
	}
}

static void bwd_prop(neural_net_t *nn, const double *target)
{
	int i, L = nn->num - 1;
	gsl_matrix_view tmp;
	tmp = gsl_matrix_view_array((double *)target, 1, nn->layers[L]);
	gsl_matrix_memcpy(nn->dlt[L-1], nn->act[L]);
	gsl_matrix_sub(nn->dlt[L-1], &tmp.matrix);
	activ_der(nn->act[L]);
	gsl_matrix_mul_elements(nn->dlt[L-1], nn->act[L]);
	for (i=L-2; i>=0; --i) {
		gsl_blas_dgemm(CblasNoTrans, CblasTrans, \
		1, nn->dlt[i+1], nn->wts[i+1], 0, nn->dlt[i]);
		activ_der(nn->act[i+1]);
		gsl_matrix_mul_elements(nn->dlt[i], nn->act[i+1]);
	}
}

static void wts_update(neural_net_t *nn)
{
	int i;
	for (i=0; i < nn->num-1; ++i) {
		gsl_blas_dgemm(CblasTrans, CblasNoTrans, \
		1, nn->act[i], nn->dlt[i], 0, nn->dwt[i]);
		gsl_matrix_scale(nn->dwt[i], nn->step);
		gsl_matrix_sub(nn->wts[i], nn->dwt[i]);
	}
}

static neural_net_t *init_mem(const int num)
{
	neural_net_t *nn = (neural_net_t *)malloc(sizeof(neural_net_t));
	if (nn == NULL) {
		perror("neural network could not be created!");
		exit(EXIT_FAILURE);
	}
	nn->wts = (gsl_matrix **)calloc(num-1, sizeof(gsl_matrix *));
	nn->dwt = (gsl_matrix **)calloc(num-1, sizeof(gsl_matrix *));
	nn->dlt = (gsl_matrix **)calloc(num-1, sizeof(gsl_matrix *));
	nn->act = (gsl_matrix **)calloc(num, sizeof(gsl_matrix *));	
	return nn;
}

neural_net_t *nn_create(
	const int *layers,
	const int num,
	const double step,
	const double range)
{
	int i, rows, cols;
	neural_net_t *nn = init_mem(num);
	nn->layers = layers;
	nn->num = num;
	nn->step = step;
	nn->range = range;
	init_rand(nn);
	for (i=0; i < num-1; ++i) {
		rows = layers[i] + 1;
		cols = (i < num-2) ? (layers[i+1] + 1) : layers[i+1];
		nn->wts[i] = gsl_matrix_alloc(rows, cols);
		wts_rand(nn, i);
		nn->dwt[i] = gsl_matrix_alloc(rows, cols);
		nn->dlt[i] = gsl_matrix_alloc(1, cols);
	}
	for (i=0; i<num; ++i) {
		cols = (i < num-1) ? (layers[i] + 1) : layers[i];
		nn->act[i] = gsl_matrix_alloc(1, cols);
	}
	return nn;
}

void nn_destroy(neural_net_t *nn)
{
	int i;
	for (i=0; i < nn->num; ++i)
		gsl_matrix_free(nn->act[i]);
	for (i=0; i < nn->num-1; ++i) {
		gsl_matrix_free(nn->dlt[i]);
		gsl_matrix_free(nn->dwt[i]);
		gsl_matrix_free(nn->wts[i]);
	}
	gsl_rng_free(nn->rng);
	free(nn->act);
	free(nn->dlt);
	free(nn->dwt);
	free(nn->wts);
	free(nn);
}

void nn_train(
	neural_net_t *nn,
	const gsl_matrix *in,
	const gsl_matrix *tgt,
	const int epochs)
{
	int i, r;
	gsl_matrix_view tmp;
	gsl_matrix *trng_set = gsl_matrix_alloc(in->size1, in->size2 + 1);
	matrix_append(trng_set, in);
	for (i=0; i<epochs; ++i) {
		r = gsl_rng_uniform_int(nn->rng, in->size1);
		tmp = gsl_matrix_submatrix(trng_set, r, 0, 1, in->size2 + 1);
		gsl_matrix_memcpy(nn->act[0], &tmp.matrix);
		fwd_prop(nn);
		bwd_prop(nn, &tgt->data[r * tgt->size2]);
		wts_update(nn);
	}
	gsl_matrix_free(trng_set);
}

void nn_predict(
	neural_net_t *nn,
	const gsl_matrix *in,
	const gsl_matrix *res)
{
	int i;
	for (i=0; i< in->size2; ++i)
		gsl_matrix_set(nn->act[0], 0, i, in->data[i]);
	gsl_matrix_set(nn->act[0], 0, i, 1);
	fwd_prop(nn);
	for (i=0; i< res->size2; ++i)
		res->data[i] = gsl_matrix_get(nn->act[nn->num-1], 0, i);
}
