#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
#include <armadillo>
using namespace cv;
using namespace std;

int lowT = 0, highT = 0, dilation_size = 3;
int lowH, lowS, lowV, highH, highS, highV;
Mat applyThreshold(Mat image, Scalar HSVlow, Scalar HSVhigh,int cannyLOW,int cannyHIGH){
	Mat cannyImage; 
	cvtColor(image, cannyImage, CV_BGR2GRAY);
	Canny(cannyImage, cannyImage, cannyLOW, cannyHIGH, 3);
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * dilation_size + 1, 2 * dilation_size + 1), Point(dilation_size, dilation_size));
	cv::dilate(cannyImage, cannyImage, element);
	//imshow("canny",cannyImage);

	Mat hsvImage;
	cvtColor(image, hsvImage, CV_BGR2HSV);
	cv::inRange(hsvImage, HSVlow, HSVhigh, hsvImage);
	element = getStructuringElement(MORPH_ELLIPSE, Size(2 * dilation_size + 1, 2 * dilation_size + 1), Point(dilation_size, dilation_size));
	cv::dilate(hsvImage, hsvImage, element);
	//imshow("hsv", hsvImage);

	Mat combined; int erosin_size = 3;
	cv::bitwise_or(cannyImage, hsvImage, combined);
	element = getStructuringElement(MORPH_ELLIPSE, Size(2 * erosin_size + 1, 2 * erosin_size + 1), Point(erosin_size, erosin_size));
	cv::erode(combined, combined, element);
	//imshow("combined", combined);
	//waitKey(100000);

	return combined;
}


void on_trackbar(int, void*){
	FILE *f = fopen("AllParams.txt", "w");
	if (!f)return;
	else{
		fprintf(f, "%d %d %d\n", lowT, highT, dilation_size);
		fprintf(f, "%d %d %d %d %d %d\n", lowH, lowS, lowV,highH,highS,highV);
		fclose(f);
	}
	
}

void createTrackBars(){
	namedWindow("colorFilter", 0);
	createTrackbar("lowH", "colorFilter", &lowH, 255, on_trackbar);
	createTrackbar("lowS", "colorFilter", &lowS, 255, on_trackbar);
	createTrackbar("lowV", "colorFilter", &lowV, 255, on_trackbar);
	createTrackbar("highH", "colorFilter", &highH, 255, on_trackbar);
	createTrackbar("highS", "colorFilter", &highS, 255, on_trackbar);
	createTrackbar("highV", "colorFilter", &highV, 255, on_trackbar);
	createTrackbar("lowT", "colorFilter", &lowT, 400, on_trackbar);
	createTrackbar("highT", "colorFilter", &highT, 400, on_trackbar);
	createTrackbar("dillat", "colorFilter", &dilation_size, 8, on_trackbar);
}

void loadAllParams(){
	FILE* f = fopen("AllParams.txt", "r");
	if (!f)return;
	else{
		fscanf(f, "%d %d %d", &lowT, &highT, &dilation_size);
		fscanf(f, "%d %d %d %d %d %d\n", &lowH, &lowS, &lowV, &highH, &highS, &highV);
		fclose(f);
	}
}

Mat region_of_interest(Mat combinedBinaryImage){
	Point points[1][4];
	points[0][0] = Point(0, combinedBinaryImage.size[0]);
	points[0][1] = Point(550,470);
	points[0][2] = Point(700, 470);
	points[0][3] = Point(combinedBinaryImage.size[1], combinedBinaryImage.size[0]);
	const Point* ppt[1] = { points[0] };
	int npt[] = { 4 };
	//Mat black(720, 1280, CV_8UC3, Scalar(0, 0, 0));
	Mat mask(720, 1280, CV_8UC1, Scalar(0));
	fillPoly(mask, ppt, npt, 1, Scalar(255, 255, 255), 8);

	Mat roi;
	cv::bitwise_and(combinedBinaryImage, mask,roi);
	return roi;
}

void getTranformationmatrix(Mat &M,Mat &Minv){
	cv::Point2f src[] = { Point2f(120,720), Point2f(550,470), Point2f(700,470), Point2f(1160,720) };
	cv::Point2f dst[] = { Point2f(200,720), Point2f(200,0), Point2f(1080,0), Point2f(1080,720) };
	M = cv::getPerspectiveTransform(src, dst);
	Minv = cv::getPerspectiveTransform(dst, src);
}

void histogram_pixels(vector<int>&leftx, vector<int>&lefty, vector<int>&rightx, vector<int>&righty, Mat &warpedImage){
	int numberOfSteps = 8;
	int pixelsPerHistogram = warpedImage.size[0]/numberOfSteps;
	int numberOfSlides = 30;
	//starting from top to bottom
	for (int i = 0; i < numberOfSteps; i++){
		int startY = i*pixelsPerHistogram;
		int endY = (i+1)*pixelsPerHistogram;

		//in order to find the center of the histogram i use i sliding window
		int slidingWindowSize = warpedImage.size[1] / numberOfSlides;
		int centerOfHistogramX = 0;
		int countOfNonEmptyWindows = 0;
		for (int j = 0; j < numberOfSlides; j++){
			int startX = j*slidingWindowSize;
			int endX = (j + 1)*slidingWindowSize;

			int meanXforSlidingWindow = 0;//basicaly the center for the cluster of points found in the sliding window
			int countOfPoints = 0; 
			for (int k = startY; k < endY; k++){
				for (int l = startX; l< endX; l++){
					if (warpedImage.at<uchar>(k, l) != 0){
						meanXforSlidingWindow += l; //this is L not 1
						countOfPoints++;
					}
				}
			}

			if (meanXforSlidingWindow == 0 || countOfPoints == 0){//meanXforSlidingWindow = (startX + endX) / 2;
				continue;
			}
			else{
				meanXforSlidingWindow /= countOfPoints;
				centerOfHistogramX += meanXforSlidingWindow;
				countOfNonEmptyWindows++;
			}
		}
		if (centerOfHistogramX == 0 || countOfNonEmptyWindows <= 1){ //if it is empty or it has only left or right lane
			centerOfHistogramX = warpedImage.size[1]/2;//eventually the histogram is ignore since it just does not contain points 
		}
		else{
			centerOfHistogramX /= countOfNonEmptyWindows;
		}

		//in order to ignore distant lanes i will start the search from the center of the histogram and move to the left and to the right and stop at the first occurence
		//first left
		for (int j = 0; j < numberOfSlides; j++){//10 is just a random number for the number of slides to search to the left
			int startX = centerOfHistogramX - (j + 1)*slidingWindowSize;
			int endX = centerOfHistogramX - j*slidingWindowSize;
			if (startX < 0 || endX < 0)break;
			int countOfPoints = 0;

			for (int k = startY; k < endY; k++){
				for (int l = startX; l < endX; l++){
					if (warpedImage.at<uchar>(k, l) != 0){
						leftx.push_back(l);
						lefty.push_back(k);
						countOfPoints++;
					}
				}
			}
			//if (countOfPoints > 0)break;
		}

		//now right
		for (int j = 0; j < numberOfSlides; j++){//10 is just a random number for the number of slides to search to the right
			int startX = centerOfHistogramX + j*slidingWindowSize;
			int endX = centerOfHistogramX + (j + 1)*slidingWindowSize;
			if (startX > warpedImage.size[1] || endX > warpedImage.size[1])break;

			int countOfPoints = 0;
			for (int k = startY; k < endY; k++){
				for (int l = startX; l < endX; l++){
					if (warpedImage.at<uchar>(k, l) != 0){
						rightx.push_back(l);
						righty.push_back(k);
						countOfPoints++;
					}
				}
			}
			//if (countOfPoints > 0)break;
		}
	}
}

int poly(int Yindep, arma::vec coefs){
	int Xdep = pow((double)coefs[0] * (double)Yindep, 2.0) + (double)coefs[1] * (double)Yindep + (double)coefs[2];
	return Xdep;
}

void drawPoly(Mat &img, arma::vec coefs){
	int height = img.size[0];
	int pixels_per_step = height / 15; //number of steps
	for (int i = 0; i < 15; i++){
		int startY = i*pixels_per_step;
		int endY = startY + pixels_per_step;
		
		Point start_line(poly(startY,coefs), startY);
		Point end_line(poly(endY, coefs), endY);
		cout << start_line << endl;
		cout << end_line << endl;
		line(img, start_line, end_line, Scalar(255, 0, 0), 1, 8);
	}
}

void drawSurface(Mat &img, arma::vec coefsleft,arma::vec coefsright){
	int height = img.size[0];
	int pixels_per_step = height / 15;
	Point points[1][32];
	for (int i = 0; i <= 15; i++){
		int startY = i*pixels_per_step;
		points[0][i] = Point(poly(startY, coefsleft), startY);
	}
	for (int i = 0; i <= 15; i++){
		int startY = height - i*pixels_per_step;
		points[0][i + 16] = Point(poly(startY, coefsright), startY);
	}

	const Point* ppt[1] = { points[0] };
	int npt[] = { 32 };

	fillPoly(img,ppt,npt,1,Scalar(255, 0, 0),8);
}

void plotPoints(Mat& img,vector<int>x,vector<int>y,Scalar color){
	for (int i = 0; i < x.size(); i++){
		circle(img, Point(x[i], y[i]), 1, color, 1, 8);
	}
}

void main(){
	loadAllParams();
	createTrackBars();
	
	Mat image = imread("quiz.png");
	int half_frame = image.size[1];
	int steps = 8;
	int pixels_per_step = image.size[0] / 2;
	Mat combined = applyThreshold(image, Scalar(lowH, lowS, lowV), Scalar(highH, highS, highV), lowT, highT);
	Mat region = region_of_interest(combined);

	Mat M, Minv; getTranformationmatrix(M,Minv);
	Mat warped; warpPerspective(region, warped, M, combined.size());

	vector<int>leftx, lefty, rightx, righty;
	histogram_pixels(leftx, lefty, rightx, righty,warped);
	arma::vec leftX; leftX = arma::conv_to<arma::vec>::from(leftx);
	arma::vec leftY; leftY = arma::conv_to<arma::vec>::from(lefty);
	arma::vec rightX; rightX = arma::conv_to<arma::vec>::from(rightx);
	arma::vec rightY; rightY = arma::conv_to<arma::vec>::from(righty);
	arma::vec PcoefsLeft = arma::polyfit(leftY, leftX,2); //version 7.7 does have polyfit
	arma::vec PcoefsRight = arma::polyfit(rightY, rightX, 2);
	PcoefsLeft.print();
	PcoefsRight.print();
	cvtColor(warped, warped, CV_GRAY2BGR);
	drawPoly(warped, PcoefsLeft);
	drawPoly(warped, PcoefsRight);
	plotPoints(warped, leftx, lefty, Scalar(0, 255, 0));
	plotPoints(warped, rightx, righty, Scalar(0, 0, 255));
	drawSurface(warped, PcoefsLeft, PcoefsRight);
	
	warpPerspective(warped, warped, Minv, combined.size());
	cv::add(image, warped, image);
	imshow("out", image );
	waitKey(1000000);

}