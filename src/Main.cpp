#include <iostream>
#include <fstream>
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\ml.hpp>
#include "HOG\hog.h"
#include "features.h"
#include "trainer.h"



#define ALLOCATIONFAULT -666
#define TEMPLATEFAILUREWIDTH -20
#define TEMPLATEFAILUREHEIGHT -21

#define TEMPLATEWIDTH 64
#define TEMPLATEHEIGHT 128
#define LAMDA 5
#define DISVALUETRESHOLD -3.0

#define POSFILE "pos.lst"
#define NEGFILE "neg.lst"

//should be 2^n for better hog aggregation
#define CELLSIZE 8


using namespace std;
using namespace cv;
vector<double***> generatePositivTrainingData(String path);
vector<double***> generateNegativTrainingsData(String path);
void slideOverImage(Mat img, string svmModel);
double*** getHOGFeatureArrayOnScaleAt(int x, int y, vector<int> &dims, double *** featArray) throw (int);
void freeHoGFeaturesOnScale(double*** feat);
void freeVectorizedFeatureArray(double ** v_feat);

int main() {
	Mat img = imread("pos_ped.jpg");
	
	//vector<double***> neg = generateNegativTrainingsData(NEGFILE);
	//vector<double***> pos = generatePositivTrainingData(POSFILE);
	//cout << "pos: " << pos.size() << endl << "neg: " << neg.size() << endl;

	//Mat train = Mat(neg);
	
	// SVM Part Test works as expected

	vector<int> pos_feat_dims;
	vector<int> neg_feat_dims;
	double ** pos_datasetFeatArray; 
	double ** neg_datasetFeatArray;
	cv::Mat responses;
	

	//get_HoG_feat_trainSets(pos_datasetFeatArray, POSFILE, CELLSIZE, TEMPLATEWIDTH,TEMPLATEHEIGHT, pos_feat_dims, true);

	//get_HoG_feat_trainSets(neg_datasetFeatArray, NEGFILE, CELLSIZE, TEMPLATEWIDTH, TEMPLATEHEIGHT, neg_feat_dims, false);

	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.C = 1;
	params.term_crit = TermCriteria(CV_TERMCRIT_EPS, 50, 0.000001);
	//train_classifier(pos_datasetFeatArray, neg_datasetFeatArray, pos_feat_dims, neg_feat_dims, svmModel, params);
	string svmModel = "svm2.xml";
	slideOverImage(img, svmModel);

	//getchar();
	return 0;
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
void slideOverImage(Mat img, string svm_model_path) {
	Mat src;
	Mat pyrtemp;
	bool show = false;
	img.copyTo(pyrtemp);
	img.copyTo(src);
	int imgheight = img.size().height;
	int imgwidth = img.size().width;
	int stage = 0;
	int gausStage = 0;
	int templateh=TEMPLATEHEIGHT;
	int templatew = TEMPLATEWIDTH;
	static double scalingfactor = pow(2, 1.0 / LAMDA);
	while (img.cols > TEMPLATEWIDTH && img.rows > TEMPLATEHEIGHT) {
		imshow("Template",src);
		waitKey(1);
		vector<int> dims;

		//********* added **********
		vector<int> vec_feat_dims;
		double scale = pow(scalingfactor, stage);
		bool person = false;
		// **************************************

		double *** featArray = computeHoG(img, CELLSIZE, dims);
		cout << dims[0] << ":" << dims[1] << ":" << dims[2] << endl;
		for (int y = CELLSIZE; y < imgheight-TEMPLATEHEIGHT; y+=CELLSIZE) {
			for (int x = CELLSIZE; x < imgwidth-TEMPLATEWIDTH; x+=CELLSIZE) {
				//x,y for HOGfeature in Template
				try	{
					double *** feat = getHOGFeatureArrayOnScaleAt(x, y, dims, featArray);

					// Predict if pedestrian stands in at this position and scale
					double ** vec_featArray = vectorize_32_HoG_feature(feat,CELLSIZE,TEMPLATEWIDTH,TEMPLATEHEIGHT,vec_feat_dims);
					//generate_SVM_predictDataSet(vec_featArray, vec_feat_dims);
					float disVal = predict_pedestrian(vec_featArray, vec_feat_dims, svm_model_path, x, y, scale, person);
					if (person == true && disVal < DISVALUETRESHOLD) {
						cout << "Found Pedestrain, distance: "<< disVal << endl;
						person = false;
						show = true;
					}
					freeHoGFeaturesOnScale(feat);
					freeVectorizedFeatureArray(vec_featArray);
				}
				catch (int n) {
					/*              +++DEBUG+++
					if (n == TEMPLATEFAILUREHEIGHT)
						
						cout << "HEIGHTERROR" << endl;
					if (n == TEMPLATEFAILUREWIDTH)
						cout << "WIDTHERROR" << endl;*/
					continue;
				}

				//size of Template in Original Window, may be needed in Future.
				if (show) {
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
					show = false;
					waitKey();
				}

				
			}
		}
		//downsample

		//imshow("Test", img);
		//waitKey();

		int w = floor(img.cols / pow(2, 1.0 / LAMDA));
		int h = floor(img.rows / pow(2, 1.0 / LAMDA));

		
		//Using GausPyramid for every Octave
		if (++gausStage < LAMDA) {
			resize(img, img, Size(w, h));
		}
		else {
			pyrDown(pyrtemp, pyrtemp);
			pyrtemp.copyTo(img);
			gausStage = 0;
		}
		imgheight = img.size().height;
		imgwidth = img.size().width;
		stage++;

		cout << "Width: " << img.size().width << " -- Height: " << img.size().height << endl;
	}
}


void freeHoGFeaturesOnScale(double*** feat) {
	int featH = TEMPLATEHEIGHT / CELLSIZE;
	int featW = TEMPLATEWIDTH / CELLSIZE;
	for (int i = 0; i < featH; i++) {
		delete[] feat[i];
	}
	delete[] feat;
	
}

void freeVectorizedFeatureArray(double ** v_feat) {
	int x_cells = (TEMPLATEWIDTH / CELLSIZE);
	int y_cells = (TEMPLATEHEIGHT / CELLSIZE);
	int num_dims = 32;
	int features = y_cells * x_cells * num_dims;
	free(v_feat[0]);
	free(v_feat);
}