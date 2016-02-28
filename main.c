#include "main.h"

int main(void)
{
	printf("[ neural network ]\n");

	int i, num, epochs;
	double step, range;
	gsl_matrix *in=NULL, *tgt=NULL, *x=NULL, *res=NULL;
	neural_net_t *nn=NULL;

	/* neural network parameters */
	num = SIZE(layers);			/* number of layers */
	step = 0.1;				/* weights scale step */
	range = 2;				/* random number range */
	epochs = 10000;				/* training iterations */

	/* convert training inputs array to gsl matrix */
	in = arr_to_gslmat(in_arr, 4, 2);
	/* convert training targets array to gsl matrix */
	tgt = arr_to_gslmat(tgt_arr, 4, 1);
	/* allocate memory for neural network input matrix */
	x = gsl_matrix_alloc(1, in->size2);
	/* allocate memory for neural network prediction result */
	res = gsl_matrix_alloc(1, layers[num-1]);

	/* create neural network */
	nn = nn_create(layers, num, step, range);

	/* train neural network */
	nn_train(nn, in, tgt, epochs);

	/* loop over training inputs  */
	for (i=0; i < in->size1; ++i) {
		set_x(in, x, i);		/* set neural network input  */
		nn_predict(nn, x, res);		/* neural network prediction */
		disp_res(in, res, i);		/* display prediction result */
	}

	/* destroy neural network */
	nn_destroy(nn);

	return EXIT_SUCCESS;
}

/* display neural nertwork prediction result matrix */
static void disp_res(
	const gsl_matrix *in,
	const gsl_matrix *res,
	const int set)
{
	int i;
	double in1, in2;
	i = in->size2 * set;
	in1 = in->data[i];			/* training input 1 */
	in2 = in->data[i+1];			/* training input 2 */
	printf("(%f, %f) -> ", in1, in2);
	for (i=0; i < res->size2-1; ++i)
		printf("%f, ", res->data[i]);
	printf("%f\n", res->data[i]);
}

/* set neural network input matrix */
static void set_x(
	const gsl_matrix *in,
	const gsl_matrix *x,
	const int set)
{
	int i;
	for (i=0; i < in->size2; ++i)
		x->data[i] = in->data[set * in->size2 + i];
}
