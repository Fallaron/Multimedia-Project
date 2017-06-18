#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>
#include "HOG\hog.h"

#define ALLOCATIONFAULT -666
#define TEMPLATEFAILUREWIDTH -20
#define TEMPLATEFAILUREHEIGHT -21

#define TEMPLATEWIDTH 64
#define TEMPLATEHEIGHT 128
#define ALPHA 5
//should be 2^n for better hog aggregation
#define CELLSIZE 8
using namespace std;
using namespace cv;

void slideOverImage(Mat img);
double*** getHOGFeatureArrayOnScaleAt(int x, int y, vector<int> &dims, double *** featArray) throw (int);

int main() {
	Mat img = imread("lenna.png");
	slideOverImage(img);

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
					Scalar green(0, 255, 0);

					int templatew = TEMPLATEWIDTH;
					templatew *= scale;
					int templateh = TEMPLATEHEIGHT;
					templateh *= scale;
					int newx = x* scale;
					int newy = y* scale;
					Mat copy;
					src.copyTo(copy);
					Point tl(newx, newy);
					Point br(newx + templatew, newy + templateh);
					//Point bl(newx + templatew,newy);
					//Point tr(newx, newy + templateh);
					rectangle(copy, tl, br, green);

					imshow("Template", copy);
					waitKey();
					

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