#ifndef NEURAL_ARMADILLO_H
#define NEURAL_ARMADILLO_H

#include <armadillo>
#include <vector>
#include "Cost.h"

using namespace std;

class Neural{

public:
	arma::Mat<int> sizes;
	int numberOfLayers;
	vector<arma::Mat<double>> biases;// vector of matrices : if sizes are (4 3 2) thhere will be 2 biases matrices. first with size(2,1) second with size (2,1)
	vector <arma::Mat<double>> weights; //vector of matrices: if sizes are (4 2 1) there will be 2 weights matrices. first with size(
	Cost *cost;

public:
	Neural();
	Neural(arma::Mat<int> &sizes);
    Neural(const Neural& n);
	~Neural();

	void defaultWeightInitializer();
	void largeWeightInitializer(); 

	arma::Mat<double> feedForward(arma::Mat<double> a);
	double SGD(vector<arma::Mat<double>> &training_data,
               vector<arma::Mat<double>> &validation_data,
		int epochs,
		int mini_batch_size,
        double eta,
		double lmbda,
		bool monitor_training_cost,
		bool monitor_training_accuracy,
		bool monitor_evaluation_cost,
		bool monitor_evaluation_accuracy );
	void update_mini_batch(vector<arma::Mat<double>> & mini_batch, double eta, double lmbda, int n);
	void backprop(arma::Mat<double> &x, arma::Mat<double> &y,vector<arma::Mat<double>> &delta_nabla_b,vector<arma::Mat<double>> &delta_nabla_w);
    arma::Mat<double> outerProduct(arma::Mat<double> & vect1, arma::Mat<double> const &vect2);
    static arma::Mat<double> sigmoid(arma::Mat<double> const &z);
    static arma::Mat<double> sigmoid_prime(arma::Mat<double> const &z);

	double total_cost(vector<arma::Mat<double>> &data, double lmbda );
	int accuracy(vector<arma::Mat<double>> &data);
    static int myrandom(int i);
};

#endif
