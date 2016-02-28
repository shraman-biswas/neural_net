#include "main.h"

int main(void)
{
	printf("[ neural network ]\n");

	int i, num_layers, num_targets;
	num_layers = SIZE(layers);
	num_targets = SIZE(tgt_arr);

	gsl_matrix *in = arr_to_gslmat(in_arr, 4, 2);
	gsl_matrix *tgt = arr_to_gslmat(tgt_arr, 4, 1);

	gsl_matrix *x = gsl_matrix_alloc(1, in->size2);
	gsl_matrix *res = gsl_matrix_alloc(1, layers[num_layers-1]);

	//nn_init(layers, num_layers, 0.1, 2);

	//nn_train(&in, &tgt, EPOCHS);

	for (i=0; i<num_targets; ++i) {
		set_x(in, x, i);
		//nn_predict(&x, &res);
		disp_res(in, res, i);
	}

	//nn_clear();

	return EXIT_SUCCESS;
}

static gsl_matrix *arr_to_gslmat(
	const double *arr,
	const int rows,
	const int cols)
{
	int i;
	gsl_matrix *m = gsl_matrix_alloc(rows, cols);
	for (i=0; i < rows * cols; ++i)
		m->data[i] = arr[i];
	return m;
}

static void disp_res(
	const gsl_matrix *in,
	const gsl_matrix *res,
	const int set)
{
	int i;
	double in1, in2;
	i = in->size2 * set;
	in1 = in->data[i];
	in2 = in->data[i+1];
	printf("(%f, %f) -> ", in1, in2);
	for (i=0; i < res->size2-1; ++i)
		printf("%f, ", res->data[i]);
	printf("%f\n", res->data[i]);
}

static void set_x(
	const gsl_matrix *in,
	const gsl_matrix *x,
	const int set)
{
	int i;
	for (i=0; i < in->size2; ++i)
		x->data[i] = in->data[set * in->size2 + i];
}
