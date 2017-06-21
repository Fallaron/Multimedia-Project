
#include "trainer.h"
#include <iostream>

using namespace cv;
using namespace std;

// generate SVM train vector set from HOG features both positive and negative..
cv::Mat generate_SVM_trainSet(double **pos_featArray, double** neg_featArray, std::vector<int> pos_dims, std::vector<int>neg_dims, cv::Mat &responses) {
	int pos_count = pos_dims[0]; //sizeof(pos_featArray) / sizeof(pos_featArray[0]);
	int neg_count = neg_dims[0]; //sizeof(neg_featArray) / sizeof(neg_featArray[0]);
	int img_total_count = pos_count + neg_count;
	int features = pos_dims[1];	//sizeof(pos_featArray[1]) / pos_count;

	cv::Mat trainDataSet = cv::Mat(img_total_count, features, CV_32FC1);
	int img = 0;
	int neg = 0;
	while (img < img_total_count) {
		if (img < pos_count) {
			for (int n = 0; n < features; n++) {
				trainDataSet.at<float>(img, n) = pos_featArray[img][n];
			}
		}
		else {
			for (int n = 0; n < features; n++) {
				trainDataSet.at<float>(img, n) = neg_featArray[neg][n];
			}
			neg++;
		}
		img++;
	}
	// label Dataset
	responses = cv::Mat(img_total_count, 1, CV_32FC1);

	int label_count = 0;
	while (label_count < img_total_count) {
		if (label_count < pos_count) {
			responses.at<float>(label_count, 0) = 1;
			//label_count++;
		}
		else {
			responses.at<float>(label_count, 0) = -1;
		}
		label_count++;
	}
	return trainDataSet;
}

std::string train_classifier(double **pos_featArray, double** neg_featArray, std::vector<int> pos_dims, std::vector<int>neg_dims, std::string SVMModel_Name) {

	Mat responses = Mat();

	Mat data = generate_SVM_trainSet(pos_featArray, neg_featArray, pos_dims, neg_dims, responses);

	cv::Mat img_sample(1, data.cols, CV_32FC1);
	cv::Mat img_response(1, 1, CV_32FC1);

	Mat sample = Mat();
	Mat varidx = Mat();
	CvSVM SVM;

	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.C = 1;
	params.term_crit = TermCriteria(CV_TERMCRIT_EPS, 50, 6e-10);
	// pick one image at a time from Mat Data and train the SVM with it and move on to the next till all are done
	bool model;

	/*for (int i = 0; i < data.rows; i++) {
	for (int n = 0; n < data.cols; n++) {
	img_sample.at<float>(0, n) = data.at<float>(i,n);
	}
	// pick a single corresponding respones for a paticular image every time
	img_response.at<float>(i, 0) = responses.at<float>(i, 0);
	model = SVM.train_auto(img_sample, img_response, varidx, sample, params);
	} */

	model = SVM.train_auto(data, responses, varidx, sample, params);

	if (model) {
		SVM.save(SVMModel_Name.c_str());
	}
	return SVMModel_Name;
}

// convert feature or features from the scanning window to dataset useable by SVM for prediction return a Mat
cv::Mat generate_SVM_predictDataSet(double **featArray, vector<int> feat_dims) {
	int imgPatch_count = feat_dims[0];
	int features = feat_dims[1];

	cv::Mat toPredictSet(imgPatch_count, features, CV_32FC1);

	int num_patch = 0;

	while (num_patch < imgPatch_count) {
		for (int n = 0; n < features; n++) {
			toPredictSet.at<float>(num_patch, n) = featArray[num_patch][n];
		}
		num_patch++;
	}
	return toPredictSet;
}

// calls the generate_predicDataset and predicts whether a pedestrian´is in or not.. this could be called inside the sliding window
// to reduce the overload of storing all possible template moves.. but rather make a template move and decide immediately if person.
// if person store the scale, x ,y values and continue scanning..
void predict_pedestrian(double ** featArray, vector<int> feat_dims, std::string svm_path) {
	CvSVM *newSVM = new CvSVM;
	newSVM->load(svm_path.c_str());

	cv::Mat predicted = generate_SVM_predictDataSet(featArray, feat_dims);
	int patchFeatures = feat_dims[1];
	cv::Mat predictSample(1, patchFeatures, CV_32FC1); // we test only one img patch at a time

													   // in case many img patches are stored in featArray go through all of them ´but predict one at a time
	for (int i = 0; i < feat_dims[0]; i++) {
		for (int j = 0; j < feat_dims[1]; j++) {
			predictSample = (cv::Mat_<float>(1, feat_dims[1]) << i, predicted.at<float>(i, j));
		}
		float res = newSVM->predict(predictSample, true);
		cout << res << endl;
	}
}
