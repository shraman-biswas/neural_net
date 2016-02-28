#include "main.h"

int main(void)
{
	printf("[ neural network ]\n");

	int i, num_layers = sizeof(layers) / sizeof(*layers);

	nn_init(layers, num_layers, 0.1, 2);

	nn_train(&in, &tgt, EPOCHS);

	double x_arr[in.cols];
	double res_arr[layers[num_layers - 1]];
	nn_matrix_t x = {1, in.cols, x_arr};
	nn_matrix_t res = {1, layers[num_layers - 1], res_arr};

	for (i=0; i<4; ++i) {
		set_x(&x, i);
		nn_predict(&x, &res);
		disp_res(&res, i);
	}

	nn_clear();

	return EXIT_SUCCESS;
}

static void disp_res(const nn_matrix_t *res, const int set)
{
	double in1, in2;
	in1 = in.data[set * in.cols];
	in2 = in.data[set * in.cols + 1];
	printf("(%f, %f) -> ", in1, in2);
	int i;
	for (i=0; i< res->cols; ++i)
		printf("%f, ", res->data[i]);
	printf("\b\b \n");
}

static void set_x(const nn_matrix_t *x, const int set)
{
	int i;
	for (i=0; i<in.cols; ++i)
		x->data[i] = in.data[set * in.cols + i];
}
