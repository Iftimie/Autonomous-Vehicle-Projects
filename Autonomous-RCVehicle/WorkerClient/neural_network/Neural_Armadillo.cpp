#include "Neural_Armadillo.h"
#include "Cost.h"
#include <armadillo>
#include <math.h>
#include <stdio.h>
#include "threads/threadtrainer.h"

Neural::Neural(arma::Mat<int> &sizes){
	this->sizes = sizes;
	this->numberOfLayers = sizes.n_cols;
	this->cost = new CrossEntropyCost();
	defaultWeightInitializer();
}

Neural::Neural(const Neural &n){
    this->sizes = n.sizes;
    this->numberOfLayers = n.sizes.n_cols;
    this->cost = new CrossEntropyCost();
    this->biases=n.biases;
    this->weights = n.weights;
}

Neural::~Neural(){
	
}

void Neural::defaultWeightInitializer(){
	this->biases = vector<arma::Mat<double>>(sizes.n_cols-1);
	for (int i = 0; i < this->sizes.n_cols-1; i++){     
		int y = this->sizes(0, i + 1);
		arma::Mat<double> column_vector(y, 1);
		column_vector.randn();
		this->biases.at(i) = column_vector; 
		//PYTHON CODE: self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
	}
	/*for (int i = 0; i < this->biases.size(); i++)
		this->biases.at(i).print();*/

	this->weights = vector<arma::Mat<double>>(sizes.n_cols - 1);
	for (int i = 0; i < this->sizes.n_cols - 1; i++){
		int x = sizes(0,i); //x cols
		int y = sizes(0, i + 1);//y rows
		arma::Mat<double> weight_matrix(y, x);
		weight_matrix.randn();
		weight_matrix = weight_matrix / sqrt(x);
		this->weights.at(i) = weight_matrix;
		//PYTHON CODE: self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
	}
	/*for (int i = 0; i < this->biases.size(); i++){
		this->weights.at(i).print(); cout << endl;
	}*/
}

void Neural::largeWeightInitializer(){
	this->biases = vector<arma::Mat<double>>(sizes.n_cols - 1);
	for (int i = 0; i < this->sizes.n_cols - 1; i++){
		int y = this->sizes(0, i + 1);
		arma::Mat<double> column_vector(y, 1);
		column_vector.randn();
		this->biases.at(i) = column_vector;
		//PYTHON CODE: self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
	}
	this->weights = vector<arma::Mat<double>>(sizes.n_cols - 1);
	for (int i = 0; i < this->sizes.n_cols - 1; i++){
		int x = sizes(0, i); //x cols
		int y = sizes(0, i + 1);//y rows
		arma::Mat<double> weight_matrix(y, x);
		weight_matrix.randn();
		this->weights.at(i) = weight_matrix;
		//PYTHON CODE: self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
	}
}

arma::Mat<double> Neural::sigmoid(arma::Mat<double> const &z){
	return 1 / (1 + arma::exp((-z)));
	//PYTHON CODE return 1.0/(1.0+np.exp(-z))
}

arma::Mat<double> Neural::sigmoid_prime(arma::Mat<double> const &z){
	return ((arma::exp((-z))) / (arma::pow(1 + arma::exp(-z), 2)));
	//PYTHON CODE return sigmoid(z)*(1-sigmoid(z))
}

arma::Mat<double> Neural::feedForward(arma::Mat<double> a){
	for (int i = 0; i < this->sizes.n_cols - 1; i++){
		arma::Mat<double> intermediar = this->weights.at(i) * a;
		arma::Mat<double> intermediar2 = intermediar + this->biases.at(i);
		a = sigmoid(intermediar2);
		//PYTHON CODE: a = sigmoid(np.dot(w, a)+b)
	}
	return a;
}

int Neural::myrandom(int i) { return std::rand() % i; }

double Neural::SGD(vector<arma::Mat<double>> &training_data,
                   vector<arma::Mat<double>> &validation_data,
	int epochs,
	int mini_batch_size,
    double eta,
	double lmbda = 0, 
	bool monitor_training_cost = false,
	bool monitor_training_accuracy = false,
	bool monitor_evaluation_cost = false,
	bool monitor_evaluation_accuracy = false){

    /*int n_data = evaluation_data.size();*/ //PYTHON CODE: if evaluation_data: n_data = len(evaluation_data)
	int n = training_data.size(); //PYTHON CODE: n = len(training_data)



    double final_cost=999;
    double last_cost=1000;
    double evaluation_cost = 10;
    double reducer = 0.9;

	srand(time(NULL));
    int j = 0;
    while(evaluation_cost <1.2 * final_cost && final_cost>ThreadTrainer::stopTreshold){ //PYTHON CODE: for j in xrange(epochs):
        j++;
		std::random_shuffle(training_data.begin(), training_data.end(), myrandom); //PYTHON CODE: random.shuffle(training_data)

        vector<vector<arma::Mat<double>>> mini_batches;
		for (int i = 0; i < n; i += mini_batch_size){
			vector<arma::Mat<double>>::const_iterator first = training_data.begin() + i;
			vector<arma::Mat<double>>::const_iterator last = training_data.begin() + i+mini_batch_size;
			vector<arma::Mat<double>> newVec(first, last);
			mini_batches.push_back(newVec);
		}//PYTHON CODE: mini_batches = [ training_data[k:k + mini_batch_size] for k in xrange(0, n, mini_batch_size)]

		for (int i = 0; i < mini_batches.size(); i++){
			update_mini_batch(mini_batches.at(i), eta, lmbda, training_data.size()); //PYTHON CODE:for mini_batch in mini_batches:self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
		}

        last_cost=final_cost;
        final_cost = this->total_cost(training_data, lmbda);//before it was cost
        if(final_cost>last_cost){
            eta=reducer*eta;
            lmbda = reducer*lmbda;
        }
        evaluation_cost = this->total_cost(validation_data,lmbda);
		if (monitor_training_cost){
            cout << "Epoch " << j << " training complete" << endl;
            cout << "Cost on training data " << final_cost<<endl;
            cout <<"Cost on validation data "<<evaluation_cost<<endl;
		}

		if (monitor_training_accuracy){
            int accuracy = this->accuracy(training_data);
            cout << "Accuracy on training data " << accuracy <<" / "<<n<< endl;
            accuracy = this->accuracy(validation_data);
            cout << "Accuracy on training data " << accuracy <<" / "<<validation_data.size()<< endl;
        }
		cout << endl;
        ThreadTrainer::saveStandardNeuralNet(*this);
	}
	
    return final_cost;
}

void Neural::update_mini_batch(vector<arma::Mat<double>> & mini_batch, double eta, double lmbda, int n){
	vector<arma::Mat<double>> nabla_b;
	vector<arma::Mat<double>> nabla_w;
	for (int i = 0; i < this->sizes.n_cols - 1; i++){
		int m = this->biases.at(i).n_rows;
		int n = this->biases.at(i).n_cols;
		nabla_b.push_back(arma::Mat<double>(m, n, arma::fill::zeros));
		m = this->weights.at(i).n_rows;
		n = this->weights.at(i).n_cols;
		nabla_w.push_back(arma::Mat<double>(m, n, arma::fill::zeros));
	}//PYTHON CODE: nabla_b = [np.zeros(b.shape) for b in self.biases] nabla_w = [np.zeros(w.shape) for w in self.weights]
	for (int i = 0; i < mini_batch.size(); i++){
		vector<arma::Mat<double>> delta_nabla_b;
		vector<arma::Mat<double>> delta_nabla_w;
		int trainingVectLenght = mini_batch.at(i).n_rows;// mini_batch.at(i).size();
		int netSizes = this->sizes.n_cols;// this->sizes.size();
		int outputVectorLength = this->sizes.at(0,netSizes - 1);
		arma::Mat<double> x = mini_batch.at(i).submat(0, 0, trainingVectLenght - outputVectorLength-1,0 ); //column vector
		arma::Mat<double> y = mini_batch.at(i).submat(trainingVectLenght - outputVectorLength, 0, trainingVectLenght-1, 0);
		this->backprop(x, y,delta_nabla_b,delta_nabla_w); //PTHON CODE:for x, y in mini_batch: delta_nabla_b, delta_nabla_w = self.backprop(x, y)
		for (int i = 0; i < nabla_b.size(); i++){
			nabla_b.at(i) = nabla_b.at(i) + delta_nabla_b.at(i);
			nabla_w.at(i) = nabla_w.at(i) + delta_nabla_w.at(i);
		}
	}

	for (int i = 0; i < this->weights.size(); i++){//PYTHON CODE: self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)] self.biases = [b - (eta / len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
		arma::Mat<double> w = this->weights.at(i);
		arma::Mat<double> nw = nabla_w.at(i);
		double normalizer1 = (1 - eta*(lmbda / n)); //PYTHON CODE: self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
		double normalizer2 = (eta / mini_batch.size());
		this->weights.at(i) = (w*normalizer1) - (nw*normalizer2);

		arma::Mat<double> b = this->biases.at(i);
		arma::Mat<double> nb = nabla_b.at(i);
		this->biases.at(i) = b - (nb*normalizer2);
	}
}

arma::Mat<double> Neural::outerProduct(arma::Mat<double> & vect1,const arma::Mat<double> &vect2){
	if (vect1.n_cols > 1 || vect2.n_rows > 1)throw 20;
	arma::Mat<double> matrix(vect1.n_rows, vect2.n_cols);
    for (unsigned int i = 0; i < vect1.n_rows; i++){
		for (int j = 0; j < vect2.n_cols; j++){
			matrix(i, j) = vect1(i, 0)*vect2(0, j);
		}
	}
	return matrix;
}

void Neural::backprop(arma::Mat<double> &x, arma::Mat<double> &y,vector<arma::Mat<double>> &delta_nabla_b,vector<arma::Mat<double>> &delta_nabla_w){
	vector<arma::Mat<double>> nabla_b;
	vector<arma::Mat<double>> nabla_w;
	for (int i = 0; i < this->sizes.n_cols - 1; i++){
		int m = this->biases.at(i).n_rows;
		int n = this->biases.at(i).n_cols;
		nabla_b.push_back(arma::Mat<double>(m, n, arma::fill::zeros));
		m = this->weights.at(i).n_rows;
		n = this->weights.at(i).n_cols;
		nabla_w.push_back(arma::Mat<double>(m, n, arma::fill::zeros));
	}
	//FEED FORWARD
	arma::Mat<double> activation = x;
	vector<arma::Mat<double>> activations; activations.push_back(activation); //# list to store all the activations, layer by layer
	vector<arma::Mat<double>> zs; //# list to store all the z vectors, layer by layer
	for (int i = 0; i < this->biases.size(); i++){ //PYTHON CODE:  for b, w in zip(self.biases, self.weights):
		arma::Mat<double> z = (this->weights.at(i) *activation) + this->biases.at(i);
		zs.push_back(z);
		activation = sigmoid(z);
		activations.push_back(activation);
	}

	//BACKWARD PASS
	int sizeZS = zs.size();
	int sizeActivations = activations.size(); //this should be the same size as number of layers
	arma::Mat<double> delta = this->cost->delta(zs.at(sizeZS-1), activations.at(sizeActivations-1), y);
	int lastIndexBiases = nabla_b.size() - 1;
	nabla_b.at(lastIndexBiases) = delta;
	int lastIndexWeights = lastIndexBiases;
	nabla_w.at(lastIndexWeights) = this->outerProduct(delta, (activations.at(sizeActivations - 2)).t());

	//if 3 layers => 3-2 =1 => for weights & biases that means the last index ---> for zs that means the same because where there is bias there is z as z = w*a+b
	//in python with 3 layers : -2 means the first z (the middle layer z=w*a+b)
	//above we resolved for the last weight and bias
	//below we resolve for the rest (starting from the tail)
	//if 3 layers (2 b and 2 w) => 1 weight and 1 bias resolved(the last) above => one more remaining weight and bias below
	//numberOfLayes -1 = number of weights and biases; -2 = correct index of last weight and bias; -3 = the correct index of second-last weight and bias
	for (int i = this->numberOfLayers - 3; i >= 0; i--){ //PYTHON CODE: for l in xrange(2, self.num_layers): z = zs[-l]
		arma::Mat<double> z = zs.at(i);
		arma::Mat<double> sp = this->sigmoid_prime(z);
		delta = (this->weights.at(i+1).t() * delta) % sp;
		nabla_b.at(i) = delta;
		nabla_w.at(i) = this->outerProduct(delta, (activations.at(i).t()));
	}
	delta_nabla_b = nabla_b;
	delta_nabla_w = nabla_w;
}

double Neural::total_cost(vector<arma::Mat<double>> &data, double lmbda){
	double cost = 0;
	for (int i = 0; i < data.size(); i++){
		int trainingVectLenght = data.at(i).size();
		int netSizes = this->sizes.size();
		int outputVectorLength = this->sizes.at(netSizes - 1);
		arma::Mat<double> x = data.at(i).submat(0, 0, trainingVectLenght - outputVectorLength -1, 0);
		arma::Mat<double> y = data.at(i).submat(trainingVectLenght - outputVectorLength, 0, trainingVectLenght-1, 0);
		arma::Mat<double> a = this->feedForward(x);
		cost += this->cost->fn(a, y) / data.size();
	}
	double sum = 0;
	for (int j = 0; j < this->weights.size(); j++){
		sum += pow(arma::norm(this->weights.at(j)), 2);
	}
	cost += 0.5*(lmbda / data.size()) *sum;
	return cost;
}

int Neural::accuracy(vector<arma::Mat<double>> &data){
	int sum = 0;
	for (int i = 0; i < data.size(); i++){
		int trainingVectLenght = data.at(i).size();
		int netSizes = this->sizes.size();
		int outputVectorLength = this->sizes.at(netSizes - 1);
		arma::Mat<double> x = data.at(i).submat(0, 0, trainingVectLenght - outputVectorLength - 1, 0);
		arma::Mat<double> y = data.at(i).submat(trainingVectLenght - outputVectorLength, 0, trainingVectLenght - 1, 0);
		
		arma::Mat<double> a = this->feedForward(x);
		int indexA = a.index_max(); //a single value colum-wise nummerastig (vectorizes and counts) if matrix
		int indexY = y.index_max();
		if (indexA == indexY)sum++;
	}
	return sum;
}
