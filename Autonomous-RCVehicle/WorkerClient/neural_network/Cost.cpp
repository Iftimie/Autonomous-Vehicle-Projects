#include "Neural_Armadillo.h"
#include "Cost.h"
#include <armadillo>



double QuadraticCost::fn(arma::Mat<double> &a, arma::Mat<double> &y){
	arma::Mat<double> result = a - y;
	return arma::dot(result, result)*0.5;
	//PYTHON CODE return 0.5*np.linalg.norm(a-y)**2
}

arma::Mat<double> QuadraticCost::delta(arma::Mat<double> &z, arma::Mat<double> &a, arma::Mat<double> &y){
	return (a - y) % Neural::sigmoid_prime(z);
	//PYTHON CODE return (a-y) * sigmoid_prime(z)
}

double CrossEntropyCost::fn(arma::Mat<double> &a, arma::Mat<double> &y){
	arma::Mat<double> result = ((-y) % (arma::log(a))) - ((1 - y) % log((1 - a)));
	result.replace(arma::datum::nan, 0);
	return arma::accu(result);
	//PYTHON CODE return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
}

arma::Mat<double> CrossEntropyCost::delta(arma::Mat<double> &z, arma::Mat<double> &a, arma::Mat<double> &y){
	return a - y;
	//PYTHON CODE return (a - y)
}