#ifndef COST_H
#define COST_H

#include <armadillo>

class Cost{
public:
	virtual double fn(arma::Mat<double> &a, arma::Mat<double> &y)=0;
	virtual arma::Mat<double> delta(arma::Mat<double> &z, arma::Mat<double> &a, arma::Mat<double> &y)=0;
};


class QuadraticCost:public Cost{
public:
	double fn(arma::Mat<double> &a, arma::Mat<double> &y);
	arma::Mat<double> delta(arma::Mat<double> &z, arma::Mat<double> &a, arma::Mat<double> &y);
};

class CrossEntropyCost :public Cost{
public:
	double fn(arma::Mat<double> &a, arma::Mat<double> &y);
	arma::Mat<double> delta(arma::Mat<double> &z, arma::Mat<double> &a, arma::Mat<double> &y);
};


#endif