#pragma once
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\ml\ml.hpp>

cv::Mat generate_SVM_trainSet(double **pos_featArray, double** neg_featArray, std::vector<int> pos_dims, std::vector<int>neg_dims, cv::Mat &responses);
cv::Mat generate_SVM_predictDataSet(double **featArray, std::vector<int> feat_dims);
std::string train_classifier(double **pos_featArray, double** neg_featArray, std::vector<int> pos_dims, std::vector<int>neg_dims, std::string SVMModel_Name, CvSVMParams params);
float  predict_pedestrian(double ** featArray, std::vector<int> feat_dims, std::string svm_path, int pos_x, int pos_y, int scale, bool &found_Person);
double ** vectorize_32_HoG_feature(double ***featArray, int cell_size, int temp_Width, int temp_Height, std::vector<int>& vec_feat_dims);

