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
#define DISVALUETRESHOLD -1

#define POSFILE "pos.lst"
#define NEGFILE "neg.lst"

#define POSTESTFILE "Testpos.lst"
#define NEGTESTFILE "Testneg.lst"
#define ANNOTATIONTESTFILE "Testannotations.lst"

//should be 2^n for better hog aggregation
#define CELLSIZE 8


using namespace std;
using namespace cv;

static vector<double **> vFalsePositives;

vector<double***> generatePositivTrainingData(String path);
vector<double***> generateNegativTrainingsData(String path);
void slideOverImage(Mat img, string svmModel);
double*** getHOGFeatureArrayOnScaleAt(int x, int y, vector<int> &dims, double *** featArray) throw (int);
void freeHoGFeaturesOnScale(double*** feat);
void freeVectorizedFeatureArray(double ** v_feat);
void freeHog(vector<int> dims, double *** feature_Array);
void retrainModel(String path, String SVMPath);
void addtoFalsePositives(double** T);

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
	string svmModel = "svm10000Linear.xml";

	//get_HoG_feat_trainSets(pos_datasetFeatArray, POSFILE, CELLSIZE, TEMPLATEWIDTH,TEMPLATEHEIGHT, pos_feat_dims, true);
	//cout << "pos: " << pos_feat_dims[0] << endl;
	//get_HoG_feat_trainSets(neg_datasetFeatArray, NEGFILE, CELLSIZE, TEMPLATEWIDTH, TEMPLATEHEIGHT, neg_feat_dims, false);
	//cout << "neg: " << neg_feat_dims[0] << endl;
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = TermCriteria(CV_TERMCRIT_ITER, 10000, 0.00001);
	//train_classifier(pos_datasetFeatArray, neg_datasetFeatArray, pos_feat_dims, neg_feat_dims, svmModel, params);
	
	retrainModel(NEGFILE,svmModel);

	
	return 0;
	getchar();
}


void addtoFalsePositives(double** T) {
	vFalsePositives.push_back(T);
	cout << "added, now:" << vFalsePositives.size() << endl;
}

void retrainModel(String path, String SVMPath) {
	ifstream locations;
	locations.open(path);
	String file;
	while (getline(locations, file)) {
		Mat img = imread(file);
		slideOverImage(img, SVMPath);
	}

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
	static double *** features;
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

	CvSVM *newSVM = new CvSVM;
	newSVM->load(svm_model_path.c_str());

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
		//cout << dims[0] << ":" << dims[1] << ":" << dims[2] << endl;
		for (int y = CELLSIZE; y < imgheight-TEMPLATEHEIGHT; y+=CELLSIZE) {
			for (int x = CELLSIZE; x < imgwidth-TEMPLATEWIDTH; x+=CELLSIZE) {
				//x,y for HOGfeature in Template
				try	{
					double *** feat = getHOGFeatureArrayOnScaleAt(x, y, dims, featArray);

					// Predict if pedestrian stands in at this position and scale
					double ** vec_featArray = vectorize_32_HoG_feature(feat,CELLSIZE,TEMPLATEWIDTH,TEMPLATEHEIGHT,vec_feat_dims);
					//generate_SVM_predictDataSet(vec_featArray, vec_feat_dims);
					float disVal = predict_pedestrian(vec_featArray, vec_feat_dims, newSVM, x, y, scale, person);
					if (person == true && disVal < DISVALUETRESHOLD) {
						cout << "Found Pedestrain, distance: "<< disVal << endl;
						person = false;
						show = true;
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
						int k = waitKey();
						if (k == 110) {  //n
							addtoFalsePositives(vec_featArray);
						}
						
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
			}
		}
		//downsample

		//imshow("Test", img);
		//waitKey();

		//Using GausPyramid for every Octave
		if (++gausStage < LAMDA) {
			Size imgSize(floor(img.cols / scalingfactor), floor(img.rows / scalingfactor));
			resize(img, img, imgSize);
		}
		else {
			pyrDown(pyrtemp, pyrtemp);
			pyrtemp.copyTo(img);
			gausStage = 0;
		}
		imgheight = img.size().height;
		imgwidth = img.size().width;
		stage++;
		// +++ DEBUG +++
		//cout << "Width: " << img.size().width << " -- Height: " << img.size().height << endl;
		freeHog(dims, featArray);
	}
}


void freeHog(vector<int> dims, double *** feature_Array) {
	for (int i = 0; i < dims[0];i++) {
		for (int j = 0; j < dims[1]; j++) {
			free(feature_Array[i][j]);
		}
		free(feature_Array[i]);
	}
	free(feature_Array);
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
	delete(v_feat[0]);
	delete(v_feat);
}