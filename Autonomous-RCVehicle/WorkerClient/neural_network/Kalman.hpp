
#include <armadillo>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <math.h>
#include "hungarian.hpp"
#include <qDebug>
using namespace std;


class KalmanFilter{
public:
	vector<pair<int, arma::Mat<double>>> trackedStates;
    vector<vector<cv::Point>> tracks;
	int dt = 1;  //our sampling rate
	double u = 0; // define acceleration magnitude to start
	double processNoise = 1.0;
    double tkn_x = 1;  //measurement noise in the horizontal direction(x axis).
    double tkn_y = 1;  //measurement noise in the horizontal direction(y axis).
	arma::Mat<double> Ez;//error in measurement
	arma::Mat<double> Ex;//covariance matrix
	arma::Mat<double> P;

	arma::Mat<double> A;
	arma::Mat<double> B;
	arma::Mat<double> C;

	arma::Mat<double> K;


	KalmanFilter(){
		Ez = arma::Mat<double>(2, 2);
		Ez << tkn_x << 0 << arma::endr
			<< 0 << tkn_y << arma::endr;
		Ex = arma::Mat<double>(4, 4);
		Ex << pow(dt, 4) / 4 << 0 << pow(dt, 3) / 2 << 0 << arma::endr
			<< 0 << pow(dt, 4) / 4 << 0 << pow(dt, 3) / 2 << arma::endr
			<< pow(dt, 3) / 2 << 0 << pow(dt, 2) << 0 << arma::endr
			<< 0 << pow(dt, 3) / 2 << 0 << pow(dt, 2) << arma::endr;
		Ex = Ex * pow(processNoise, 2);
		P = Ex;

		A = arma::Mat<double>(4, 4);
		A << 1 << 0 << dt << 0 << arma::endr
			<< 0 << 1 << 0 << dt << arma::endr
			<< 0 << 0 << 1 << 0 << arma::endr
			<< 0 << 0 << 0 << 1 << arma::endr;
		B = arma::Mat<double>(4, 1);
		B << pow(dt, 2) / 2 << arma::endr
			<< pow(dt, 2) / 2 << arma::endr
			<< dt << arma::endr
			<< dt << arma::endr;
		C = arma::Mat<double>(2, 4);
		C << 1 << 0 << 0 << 0 << arma::endr
			<< 0 << 1 << 0 << 0 << arma::endr;
	}

	void predict(){
		for (int i = 0; i < this->trackedStates.size(); i++){
			this->trackedStates[i].second = this->A * this->trackedStates[i].second + B * u;
		}
		this->P = A * P * A.t() + this->Ex;
		this->K = P * C.t() * arma::inv(C * P * C.t() + Ez);
		//printf("\n");
		//this->K.print();
	}

	void draw(cv::Mat &img, vector<cv::Point>& detections){
		for (int i = 0; i < detections.size(); i++){
            cv::circle(img, detections[i], 3, cv::Scalar(255, 255, 255), 1, 8, 0);
		}
		for (int i = 0; i < trackedStates.size(); i++){
            cv::Point p(trackedStates[i].second(0, 0), trackedStates[i].second(1, 0));
            cv::circle(img, p, 3, cv::Scalar(40, 190, 70), 1, 8, 0);
		}
		for (int i = 0; i < tracks.size(); i++){
			for (int j = 0; j < tracks[i].size() - 1; j++){
                cv::line(img, tracks[i][j], tracks[i][j + 1],cv:: Scalar((i * 20) % 255, 255 - (i * 20) % 255, (i * 40) % 255), 1, 8, 0);
			}
		}
	}

	void update(vector<cv::Point>& detections){
		int *cost_matrix = nullptr;
		int m, n, dummy_columns;
		createCostMatrix(cost_matrix, m, n, dummy_columns, detections);
		HungarianT prob(cost_matrix, m, n, dummy_columns, HUNGARIAN_MIN);
		prob.hungarian_solve();
        //printf("%d\n",prob.hungarian_benefit());
		//TODO prob.hungarian_print_assignment();
		vector<pair<int, int>>assignements;
		vector<int> trackingsWithoutDetections, ignoredDetections;
		prob.hungarian_print_assignment(assignements, trackingsWithoutDetections, ignoredDetections);
		//[asgn, cost] = ASSIGNMENTOPTIMAL(est_dist); %do the assignment with hungarian algo

		for (int i = 0; i < assignements.size(); i++){
			int x = assignements.at(i).first;
			int y = assignements.at(i).second;
			if (true/*cost_matrix[x* n + y] < 100*/){
				/*arma::Mat<double> detectedPoint(4, 1);
				detectedPoint << detections.at(y).x << arma::endr
				<< detections.at(y).y << arma::endr
				<< 0					<< arma::endr
				<< 0					<< arma::endr;*/
				arma::Mat<double> detectedPoint(2, 1);
				detectedPoint << detections.at(y).x << arma::endr
					<< detections.at(y).y << arma::endr;


                arma::Mat<double> trackedPoint(2,1);
                trackedPoint<<trackedStates.at(x).second(0,0)<<arma::endr
                           <<trackedStates.at(x).second(1,0);
                arma::Mat<double> difference = trackedPoint-detectedPoint;
                if(sqrt(arma::dot(difference,difference))>50)continue;


				this->trackedStates.at(x).second = this->trackedStates.at(x).second + this->K * (detectedPoint - C * this->trackedStates.at(x).second);
				//Q_estimate(:,k) = Q_estimate(:,k) + K * (Q_loc_meas(asgn(F),:)' - C * Q_estimate(:,k));
                cv::Point p(trackedStates.at(x).second(0, 0), trackedStates.at(x).second(1, 0));
				this->tracks[x].push_back(p);

                cv::Point tracked(trackedStates.at(x).second(0, 0), trackedStates.at(x).second(1, 0));

			}
			else{
                qDebug("tracking left withoud assignement");
			}
		}

        cv::waitKey(1);
		if (trackingsWithoutDetections.size() != 0){
			for (int i = 0; i < m; i++){
				for (int j = 0; j < n; j++){
					printf("%d ", cost_matrix[i*(n)+j]);
				}
				printf("\n");
			}

			for (int i = 0; i < assignements.size(); i++){
				printf("%d %d\n", assignements[i].first, assignements[i].second);
			}
			printf("\n");

			for (int i = 0; i < trackingsWithoutDetections.size(); i++){
				printf("%d\n", trackingsWithoutDetections[i]);
			}
			printf("\n");
			for (int i = 0; i < ignoredDetections.size(); i++){
				printf("%d\n", ignoredDetections[i]);
			}
		}
		free(cost_matrix);
		arma::Mat<double> I(4, 4); I.eye();
		this->P = (I - this->K*this->C)*P;

		//%give a strike to any tracking that didn't get matched up to a detection

		for (int i = 0; i < trackingsWithoutDetections.size(); i++){
			int row = trackingsWithoutDetections.at(i);
			trackedStates[row].first++;
		}

		//limit tracks size
		for (int i = 0; i < tracks.size(); i++){
			if (tracks[i].size()>30){
                vector<cv::Point>::iterator it = tracks[i].begin();
				it = tracks[i].erase(it);
			}
		}

		vector<pair<int, arma::Mat<double>>>::iterator it = trackedStates.begin();
        vector<vector<cv::Point>>::iterator itp = tracks.begin();
		while (it != trackedStates.end()) {
            if (it->first >= 8 || (it->second(0,0)>640 || it->second(0,0)<0 || it->second(1,0)>480 || it->second(1,0)<0)) {
				it = trackedStates.erase(it);
				itp = tracks.erase(itp);
			}
			else {
				++it;
				++itp;
			}
		}

		//%find the new detections. basically, anything that doesn't get assigned
		for (int i = 0; i < ignoredDetections.size(); i++){
			arma::Mat<double> detectedPoint(4, 1);
			int col = ignoredDetections.at(i);
			detectedPoint << detections.at(col).x << arma::endr
				<< detections.at(col).y << arma::endr
				<< 0 << arma::endr
				<< 0 << arma::endr;
			pair<int, arma::Mat<double>> p(0, detectedPoint);
			trackedStates.push_back(p);
            vector<cv::Point> track; track.push_back(detections.at(col));
			tracks.push_back(track);
		}

	}



	void createCostMatrix(int * &cost_matrix, int &m, int &n, int &dummy_columns, vector<cv::Point>& detections){
		m = this->trackedStates.size();
		n = detections.size();
		dummy_columns = m - n;
		if (dummy_columns < 0)dummy_columns = 0;
		if (n < m)n = m;
		cost_matrix = (int*)malloc(sizeof(int)*(m)*(n));
		for (int i = 0; i < m; i++){
			cv::Point trackedPoint(trackedStates[i].second(0, 0), trackedStates[i].second(1, 0));
			for (int j = 0; j < n - dummy_columns; j++){
				cost_matrix[i*(n)+j] = euclideanDist(trackedPoint, detections[j]);
				if (cost_matrix[i*(n)+j] == 0)cost_matrix[i*(n)+j] = 1;
				//double val = (1. / (double)cost_matrix[i*n + j]) * 1000000;
				//cost_matrix[i* n + j] = (int)val;
			}
		}
		int min = 0;
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n - dummy_columns; j++)
				if (cost_matrix[i*n + j] <min)min = cost_matrix[i*n + j];
		for (int i = 0; i < m; i++)
			for (int j = n - dummy_columns; j < n; j++)
				cost_matrix[i*n + j] = 0;

		/*for (int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
		printf("%d ", cost_matrix[i*n + j]);
		}
		printf("\n");
		}*/
	}

	float euclideanDist(cv::Point& p, cv::Point& q) {
		cv::Point diff = p - q;
		return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
	}
};
