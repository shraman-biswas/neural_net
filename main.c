#include "main.h"

int main(void)
{
	printf("[ neural network ]\n");

	int i, num, epochs;
	double step, range;
	gsl_matrix *train=NULL, *target=NULL, *test=NULL, *result=NULL;
	neural_net_t *nn=NULL;

	/* neural network parameters */
	num = SIZE(layers);			/* number of layers */
	step = 0.1;				/* weights scale step */
	range = 2;				/* random number range */
	epochs = 10000;				/* training iterations */

	/* convert training inputs array to gsl matrix */
	train = arr_to_gslmat(train_arr, 4, 2);
	/* convert training targets array to gsl matrix */
	target = arr_to_gslmat(target_arr, 4, 1);
	/* allocate memory for testing inputs matrix */
	test = gsl_matrix_alloc(1, train->size2);
	/* allocate memory for neural network prediction result */
	result = gsl_matrix_alloc(1, layers[num-1]);

	/* create neural network */
	nn = nn_create(layers, num, step, range);

	/* train neural network */
	nn_train(nn, train, target, epochs);

	/* loop over testing inputs  */
	for (i=0; i < train->size1; ++i) {
		select_test(train, test, i);	/* select testing input */
		nn_predict(nn, test, result);	/* neural network prediction */
		disp_result(train, result, i);	/* display prediction */
	}

	/* destroy neural network */
	nn_destroy(nn);

	return EXIT_SUCCESS;
}

/* display neural nertwork prediction result matrix */
static void disp_result(
	const gsl_matrix *train,
	const gsl_matrix *result,
	const int set)
{
	int i;
	double input1, input2;
	i = train->size2 * set;
	input1 = train->data[i];		/* training input 1 */
	input2 = train->data[i+1];		/* training input 2 */
	printf("(%f, %f) -> ", input1, input2);
	for (i=0; i < result->size2-1; ++i)
		printf("%f, ", result->data[i]);
	printf("%f\n", result->data[i]);
}

/* select neural network input matrix */
static void select_test(
	const gsl_matrix *train,
	const gsl_matrix *test,
	const int set)
{
	int i;
	for (i=0; i < train->size2; ++i)
		test->data[i] = train->data[set * train->size2 + i];
}
