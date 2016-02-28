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

static void rand_init(void)
{
	gsl_rng_env_setup();
	nn.rng = gsl_rng_alloc(gsl_rng_default);
}

static void matrix_set_rand(gsl_matrix *m)
{
	int i, j;
	double tmp;
	for (i=0; i < m->size1; ++i) {
		for (j=0; j < m->size2; ++j) {
			tmp = 2 * gsl_rng_uniform(nn.rng) - 1;
			gsl_matrix_set(m, i, j, tmp * nn.range);
		}
	}
}

static void matrix_append(gsl_matrix *m, const nn_matrix_t *in)
{
	int cols = in->cols;
	int i, j, tmp;
	for (i=0; i< in->rows; ++i) {
		for (j=0; j<cols+1; ++j) {
			tmp = (j < cols) ? in->data[i * cols + j] : 1;
			gsl_matrix_set(m, i, j, tmp);
		}
	}
}

static void fwd_prop(void)
{
	int i;
	for (i=1; i<nn.num; ++i) {
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, \
		1, nn.act[i-1], nn.wts[i-1], 0, nn.act[i]);
		activ(nn.act[i]);
	}
}

static void bwd_prop(const double *target)
{
	int i, L = nn.num - 1;
	gsl_matrix_view tmp;
	tmp = gsl_matrix_view_array((double *)target, 1, nn.layers[L]);
	gsl_matrix_memcpy(nn.dlt[L-1], nn.act[L]);
	gsl_matrix_sub(nn.dlt[L-1], &tmp.matrix);
	activ_der(nn.act[L]);
	gsl_matrix_mul_elements(nn.dlt[L-1], nn.act[L]);
	for (i=L-2; i>=0; --i) {
		gsl_blas_dgemm(CblasNoTrans, CblasTrans, \
		1, nn.dlt[i+1], nn.wts[i+1], 0, nn.dlt[i]);
		activ_der(nn.act[i+1]);
		gsl_matrix_mul_elements(nn.dlt[i], nn.act[i+1]);
	}
}

static void wts_update(void)
{
	int i;
	for (i=0; i < nn.num-1; ++i) {
		gsl_blas_dgemm(CblasTrans, CblasNoTrans, \
		1, nn.act[i], nn.dlt[i], 0, nn.dwt[i]);
		gsl_matrix_scale(nn.dwt[i], nn.step);
		gsl_matrix_sub(nn.wts[i], nn.dwt[i]);
	}
}

void nn_init(
	const int *layers,
	const int num,
	const double step,
	const double range)
{
	int i, rows, cols;
	nn.layers = layers;
	nn.num = num;
	nn.step = step;
	nn.range = range;
	nn.wts = (gsl_matrix **)calloc(num - 1, sizeof(gsl_matrix *));
	nn.dwt = (gsl_matrix **)calloc(num - 1, sizeof(gsl_matrix *));
	nn.dlt = (gsl_matrix **)calloc(num - 1, sizeof(gsl_matrix *));
	nn.act = (gsl_matrix **)calloc(num, sizeof(gsl_matrix *));
	rand_init();
	for (i=0; i < num-1; ++i) {
		rows = layers[i] + 1;
		cols = (i < num-2) ? (layers[i+1] + 1) : layers[i+1];
		nn.wts[i] = gsl_matrix_alloc(rows, cols);
		matrix_set_rand(nn.wts[i]);
		nn.dwt[i] = gsl_matrix_alloc(rows, cols);
		nn.dlt[i] = gsl_matrix_alloc(1, cols);
	}
	for (i=0; i<num; ++i) {
		cols = (i < num - 1) ? (layers[i] + 1) : layers[i];
		nn.act[i] = gsl_matrix_alloc(1, cols);
	}
}

void nn_clear(void)
{
	int i;
	for (i=0; i<nn.num; ++i)
		gsl_matrix_free(nn.act[i]);
	for (i=0; i < nn.num-1; ++i) {
		gsl_matrix_free(nn.dlt[i]);
		gsl_matrix_free(nn.dwt[i]);
		gsl_matrix_free(nn.wts[i]);
	}
	gsl_rng_free(nn.rng);
	free(nn.act);
	free(nn.dlt);
	free(nn.dwt);
	free(nn.wts);
}

void nn_train(const nn_matrix_t *in, const nn_matrix_t *tgt, const int epochs)
{
	int i, r, rows = in->rows, cols = in->cols;
	gsl_matrix *trng_set = gsl_matrix_alloc(rows, cols + 1);
	matrix_append(trng_set, in);
	gsl_matrix_view tmp;
	for (i=0; i<epochs; ++i) {
		r = gsl_rng_uniform_int(nn.rng, rows);
		tmp = gsl_matrix_submatrix(trng_set, r, 0, 1, cols + 1);
		gsl_matrix_memcpy(nn.act[0], &tmp.matrix);
		fwd_prop();
		bwd_prop(&tgt->data[r * tgt->cols]);
		wts_update();
	}
	gsl_matrix_free(trng_set);
}

void nn_predict(const nn_matrix_t *in, const nn_matrix_t *res)
{
	int i;
	for (i=0; i< in->cols; ++i)
		gsl_matrix_set(nn.act[0], 0, i, in->data[i]);
	gsl_matrix_set(nn.act[0], 0, i, 1);
	fwd_prop();
	for (i=0; i< res->cols; ++i)
		res->data[i] = gsl_matrix_get(nn.act[nn.num-1], 0, i);
}
