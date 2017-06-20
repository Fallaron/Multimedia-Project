#include <iostream>
#include <fstream>
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\ml.hpp>
#include "HOG\hog.h"

#define ALLOCATIONFAULT -666
#define TEMPLATEFAILUREWIDTH -20
#define TEMPLATEFAILUREHEIGHT -21

#define TEMPLATEWIDTH 64
#define TEMPLATEHEIGHT 128
#define ALPHA 5

#define POSFILE "pos.lst"
#define NEGFILE "neg.lst"

//should be 2^n for better hog aggregation
#define CELLSIZE 8


using namespace std;
using namespace cv;
vector<double***> generatePositivTrainingData(String path);
vector<double***> generateNegativTrainingsData(String path);
void slideOverImage(Mat img);
double*** getHOGFeatureArrayOnScaleAt(int x, int y, vector<int> &dims, double *** featArray) throw (int);

int main() {
	Mat img = imread("lenna.png");
	//slideOverImage(img);
	vector<double***> neg = generateNegativTrainingsData(NEGFILE);
	vector<double***> pos = generatePositivTrainingData(POSFILE);
	cout << "pos: " << pos.size() << endl << "neg: " << neg.size() << endl;

	Mat train = Mat(neg);

	getchar();
}


vector<double***> generatePositivTrainingData(String path) {
	ifstream locations;
	locations.open(path);
	String file;
	vector<double***> positivHogFeatures;
	while (getline(locations, file)) {
		Mat img = imread(file);
		vector<int> dims;
		double*** feat = computeHoG(img, CELLSIZE, dims);
		double*** positiv = getHOGFeatureArrayOnScaleAt(16, 16, dims, feat);
		positivHogFeatures.push_back(positiv);
	}
	cout << "Generated Positiv Training Hog Features" << endl;
	return positivHogFeatures;
}


vector<double***> generateNegativTrainingsData(String path) {
	ifstream locations;
	locations.open(path);
	String file;
	vector<double***> negativHogFeatures;
	int c = 0;
	while (getline(locations, file)) {
		Mat img = imread(file);
		vector<int> dims;
		double*** feat = computeHoG(img, CELLSIZE, dims);
		c = 0;
		for (int y = CELLSIZE; y < img.rows - TEMPLATEHEIGHT && c < 10; y += CELLSIZE) {
			for (int x = CELLSIZE; x < img.cols - TEMPLATEWIDTH && c < 10; x += CELLSIZE) {
				try
				{
					double *** featuresNEG = getHOGFeatureArrayOnScaleAt(x, y, dims, feat);
					negativHogFeatures.push_back(featuresNEG);
					c++;
				}
				catch (int n)
				{
					continue;
				}
			}
		}
		if (c < 10)
			cout << "FAULT";
	}
	cout << "Generated Negative Training Hog Features" << endl;
	return negativHogFeatures;
}

//scale 0 = just img;
double*** getHOGFeatureArrayOnScaleAt(int x, int y, vector<int> &dims, double *** featArray) throw (int) {
	
	int hogposW = x/CELLSIZE -1;
	int hogposH = y/CELLSIZE -1;
	int featW = TEMPLATEWIDTH / CELLSIZE;
	int featH = TEMPLATEHEIGHT / CELLSIZE;
	double *** features;
	features = new double**[featH];
	if (features == NULL) {
		delete[] features;
		throw ALLOCATIONFAULT;
	}

	for (int i = 0; i < featH; i++) {
		features[i] = new double *[featW];
		if (features[i] == NULL) {
			for (int k = 0; k < i; k++) {
				delete[] features[k];
			}
			delete[] features;
			throw ALLOCATIONFAULT;
		}
		//these will happen sometime, just continue with next window.
		//it means that the Template window tries to use hogfeatures that lie beyond the hogfeatures.
		for (int j = 0; j < featW; j++) {
			if (dims[0] <= i + hogposH)  {
				delete[] *features;
				delete[] features;
				
				throw TEMPLATEFAILUREHEIGHT;
			}
			if (dims[1] <= j + hogposW) {
				delete[] * features;
				delete[] features;

				throw TEMPLATEFAILUREWIDTH;
			}
			features[i][j] = featArray[i + hogposH][j + hogposW];
		}

	}
	return features;
}

//Slides of an Image and aggregates HoG over Template Window
void slideOverImage(Mat img) {
	Mat src;
	img.copyTo(src);
	int imgheight = img.size().height;
	int imgwidth = img.size().width;
	int stage = 0;
	int templateh=TEMPLATEHEIGHT;
	int templatew = TEMPLATEWIDTH;
	static double scalingfactor = pow(2, 1.0 / ALPHA);
	while (img.cols > TEMPLATEWIDTH && img.rows > TEMPLATEHEIGHT) {
		vector<int> dims;
		double *** featArray = computeHoG(img, CELLSIZE, dims);
		cout << dims[0] << ":" << dims[1] << ":" << dims[2] << endl;
		for (int y = CELLSIZE; y < imgheight-TEMPLATEHEIGHT; y+=CELLSIZE) {
			for (int x = CELLSIZE; x < imgwidth-TEMPLATEWIDTH; x+=CELLSIZE) {
				//x,y for HOGfeature in Template
				try
				{
					double *** feat = getHOGFeatureArrayOnScaleAt(x, y, dims, featArray);
				}
				catch (int n)
				{
					if (n == TEMPLATEFAILUREHEIGHT)
						cout << "HEIGHTERROR" << endl;
					if (n == TEMPLATEFAILUREWIDTH)
						cout << "WIDTHERROR" << endl;
					continue;
				}
				
				//size of Template in Original Window, may be needed in Future.
				if (true) {
					double scale = pow(scalingfactor, stage);

					int templatew = TEMPLATEWIDTH;
					templatew *= scale;
					int templateh = TEMPLATEHEIGHT;
					templateh *= scale;
					int newx = x* scale;
					int newy = y* scale;
					Mat copy;
					src.copyTo(copy);
					Point p1(newx, newy);
					Point p2(newx + templatew,newy);
					Point p3(newx, newy + templateh);
					Point p4(newx + templatew, newy + templateh);
					line(copy, p1, p2, Scalar(0, 255, 0));
					line(copy, p1, p3, Scalar(0, 255, 0));
					line(copy, p3, p4, Scalar(0, 255, 0));
					line(copy, p4, p2, Scalar(0, 255, 0));

					imshow("Template", copy);
					waitKey(1);
					

				}

				//Do Something with Aggregated HoG Array
			}
		}
		//downsample
		//imshow("Test", img);
		//waitKey();
		int w = floor(img.cols / pow(2, 1.0 / ALPHA));
		int h = floor(img.rows / pow(2, 1.0 / ALPHA));
		//TODO Use Gaus for each octave
		resize(img, img,Size(w,h));
		imgheight = img.size().height;
		imgwidth = img.size().width;
		stage++;

		cout << "Width: " << img.size().width << " -- Height: " << img.size().height << endl;
	}



}