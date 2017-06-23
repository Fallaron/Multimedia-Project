
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
	// label Dataset
	responses = cv::Mat(img_total_count, 1, CV_32FC1);
	int img = 0;
	int neg = 0;
	while (img < img_total_count) {
		if (img < pos_count) {
			responses.at<float>(img, 0) = 1;
			for (int n = 0; n < features; n++) {
				trainDataSet.at<float>(img, n) = pos_featArray[img][n];
			}
		}
		else {
			responses.at<float>(img, 0) = -1;
			for (int n = 0; n < features; n++) {
				trainDataSet.at<float>(img, n) = neg_featArray[neg][n];
			}
			neg++;
		}
		img++;
	}
	
	
	return trainDataSet;
}

std::string train_classifier(double **pos_featArray, double** neg_featArray, std::vector<int> pos_dims, std::vector<int>neg_dims, std::string SVMModel_Name, CvSVMParams params) {

	Mat responses;
	Mat data = generate_SVM_trainSet(pos_featArray, neg_featArray, pos_dims, neg_dims, responses);
	cv::Mat img_sample(1, data.cols, CV_32FC1);
	cv::Mat img_response(1, 1, CV_32FC1);

	Mat sample = Mat();
	Mat varidx = Mat();
	CvSVM SVM;

	
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
float  predict_pedestrian(double ** featArray, vector<int> feat_dims, CvSVM *newSVM, int pos_x, int pos_y, int scale, bool &found_Person) {
	

	cv::Mat predicted = generate_SVM_predictDataSet(featArray, feat_dims);
	int patchFeatures = feat_dims[1];
	cv::Mat predictSample(1, patchFeatures, CV_32FC1); // we test only one img patch at a time

	vector<std::vector<float>> prediction = vector<std::vector<float>>(3);

	// in case many img patches are stored in featArray go through all of them ´but predict one at a time
	int p = 0;
	for (int i = 0; i < feat_dims[0]; i++) {
		for (int j = 0; j < feat_dims[1]; j++) {
			//predictSample = (cv::Mat_<float>(i, feat_dims[1]) << i, predicted.at<float>(i, j));
			predictSample.at<float>(i, j) = predicted.at<float>(i, j);
		}
		
		// +++DEBUG+++
		//cout << res <<",";
		// store results of prediction if person
		
	}
	float res = newSVM->predict(predictSample);
	float distanceDFVal = newSVM->predict(predictSample, true);
	if (res == 1) {
		//prediction[p][0] = pos_x;
		//prediction[p][1] = pos_y;
		//prediction[p][2] = scale;
		found_Person = true;
	}
	return distanceDFVal;
}

//to be used in scanning window fucntion  to tap each single scan.. USE WITH predict_pedestrain function
double ** vectorize_32_HoG_feature(double ***featArray, int cell_size, int temp_Width, int temp_Height, vector<int>& vec_feat_dims) {

	int x_cells = (temp_Width / cell_size);
	int y_cells = (temp_Height / cell_size);
	int num_dims = 32;
	int features = y_cells * x_cells * num_dims;
	vec_feat_dims = vector<int>(2);
	vec_feat_dims[0] = 1;
	vec_feat_dims[1] = features;

	// memory for vectorized HoG feature
	double ** vectorised_featArray = (double**)malloc(1 * sizeof(double**));

		vectorised_featArray[0] = (double*)malloc(features * sizeof(double));
	

	//vectorize
	int h = 0;
	for (int i = 0; i < y_cells; i++) {
		for (int j = 0; j < x_cells; j++) {
			for (int n = 0; n < num_dims; n++) {
				vectorised_featArray[0][h++] = featArray[i][j][n];
			}
		}
	}
	return vectorised_featArray;
}