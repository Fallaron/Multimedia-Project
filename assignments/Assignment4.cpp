#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\ml.hpp>
#include <time.h>
using namespace cv;

#define RANDOMNUMBERS 30
#define HEIGHT 256
#define WIDTH 256
void createtestdata(Mat &responses, Mat &data, bool overlap);
void ex1();
void ex2();


int main() {
	srand(time(NULL));
	ex1();
	ex2();
}

void ex2(){
	Mat data = Mat();
	Mat response = Mat();
	createtestdata(response, data, true);
	Mat sample = Mat();
	Mat varidx = Mat();
	Mat sample2 = Mat();
	Mat varidx2 = Mat();
	CvSVM SVM;
	CvSVM SVMRBF;

	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.C = 1;
	params.term_crit = TermCriteria(CV_TERMCRIT_ITER, 200000, 2e-10);

	CvSVMParams paramsRBF;
	paramsRBF.svm_type = CvSVM::C_SVC;
	paramsRBF.kernel_type = CvSVM::RBF;
	paramsRBF.C = 10000;
	paramsRBF.term_crit = TermCriteria(CV_TERMCRIT_ITER, 10000, 2e-11);


	SVM.train_auto(data, response, varidx, sample, params);
	SVMRBF.train_auto(data, response, varidx2, sample2, paramsRBF);

	Mat Visual = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
	Mat VisualRBF = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);

	Vec3b green(0, 100, 0), red(0, 0, 100);
	Vec3b lighterGreen(0, 150, 0), lighterRed(0, 0, 150);
	// Show the decision regions given by the SVM
	for (int i = 0; i < Visual.rows; ++i)
		for (int j = 0; j < Visual.cols; ++j)
		{
			Mat sampleMat = (Mat_<float>(1, 2) << i, j);
			float response = SVM.predict(sampleMat);
			float responseRBF = SVMRBF.predict(sampleMat);
			float marg = SVM.predict(sampleMat, true);
			float margRBF = SVMRBF.predict(sampleMat, true);


			if (responseRBF > 0)
				VisualRBF.at<Vec3b>(i, j) = green;
			else if (responseRBF < 0)
				VisualRBF.at<Vec3b>(i, j) = red;
			if (margRBF > 0 && margRBF < 1)
				VisualRBF.at<Vec3b>(i, j) = lighterRed;
			if (margRBF < 0 && margRBF > -1)
				VisualRBF.at<Vec3b>(i, j) = lighterGreen;

			if (response > 0)
				Visual.at<Vec3b>(i, j) = green;
			else if (response < 0)
				Visual.at<Vec3b>(i, j) = red;
			if (marg > 0 && marg < 1)
				Visual.at<Vec3b>(i, j) = lighterRed;
			if (marg < 0 && marg > -1) 
				Visual.at<Vec3b>(i, j) = lighterGreen;
			
		}
	for (int i = 0; i < RANDOMNUMBERS * 2; i++) {
		int x = (int)data.at<float>(i, 1);
		int y = (int)data.at<float>(i, 0);

		if (response.at<float>(i) < 0) {
			circle(Visual, Point(x, y), 3, Scalar(0, 0, 255), -1);
			circle(VisualRBF, Point(x, y), 3, Scalar(0, 0, 255), -1);
		}
		else {
			circle(Visual, Point(x, y), 3, Scalar(0, 255, 0), -1);
			circle(VisualRBF, Point(x, y), 3, Scalar(0, 255, 0), -1);
		}
	}
	imshow("Linear", Visual);
	imshow("RBF", VisualRBF);
	imwrite("Linear.jpg", Visual);
	imwrite("RBF.jpg", VisualRBF);
	waitKey();
	destroyAllWindows();
}

void ex1() {
	Mat data = Mat();
	Mat response = Mat();
	createtestdata(response, data, false);
	Mat sample = Mat();
	Mat varidx = Mat();
	CvSVM SVM;

	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.C = 1;
	params.term_crit = TermCriteria(CV_TERMCRIT_EPS, 50, 6e-10);


	SVM.train_auto(data, response, varidx, sample, params);
	Mat Visual = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
	Vec3b green(0, 100, 0), red(0, 0, 100);
	// Show the decision regions given by the SVM
	for (int i = 0; i < Visual.rows; ++i)
		for (int j = 0; j < Visual.cols; ++j)
		{
			Mat sampleMat = (Mat_<float>(1, 2) << i, j);
			float response = SVM.predict(sampleMat);
			float marg = SVM.predict(sampleMat, true);

			if (response > 0)
				Visual.at<Vec3b>(i, j) = green;
			else if (response < 0)
				Visual.at<Vec3b>(i, j) = red;
			if (marg > 0 && marg < 1) {
				Visual.at<Vec3b>(i, j) = Vec3b(0, 0, 150);
			}
			if (marg < 0 && marg > -1) {
				Visual.at<Vec3b>(i, j) = Vec3b(0, 150, 0);
			}
		}
	for (int i = 0; i < RANDOMNUMBERS * 2; i++) {
		int x = (int)data.at<float>(i, 1);
		int y = (int)data.at<float>(i, 0);

		if (response.at<float>(i) < 0)
			circle(Visual, Point(x, y), 3, Scalar(0, 0, 255), -1);
		else
			circle(Visual, Point(x, y), 3, Scalar(0, 255, 0), -1);
	}
	imshow("Linear", Visual);
	imwrite("linearklar.jpg", Visual);
	waitKey();
	destroyAllWindows();
}

void createtestdata(Mat &responses, Mat &data, bool overlap) {
	double fak=0.9;
	if (overlap)
		fak = 1.1;
	responses = Mat::zeros(RANDOMNUMBERS * 2, 1, CV_32FC1);
	data = Mat::zeros(RANDOMNUMBERS * 2, 2, CV_32FC1);
	int i;
	for ( i = 0; i < RANDOMNUMBERS; i++) {
		int rw = rand() % WIDTH;
		int rh = rand() % (int)((HEIGHT*fak) / 2);
		responses.at<float>(i) = 1;
		data.at<float>(i, 0) = rh;
		data.at<float>(i, 1) = rw;
	}
	//negativ
	for (i; i < RANDOMNUMBERS*2; i++) {
		int rw = rand() % WIDTH;
		int rh = HEIGHT - rand() % (int)((HEIGHT*fak) / 2);

		responses.at<float>(i) = -1;
		data.at<float>(i, 0) = rh;
		data.at<float>(i, 1) = rw;

	}
}

