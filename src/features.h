#pragma once

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\ml\ml.hpp>

#include <iostream>
#include <String>
#include <fstream>

void  getBoundingBox(std::string annotationList, std::vector<std::vector<int>>& boundingBoxes);
std::vector<int> detection_true_count(const std::vector<std::vector<float>> prediction_bBox,const std::vector<int> boundingBoxes);
void get_HoG_feat_trainSets(double **&dataSet_featArray, std::string dataSet_path, const int cell_size, std::vector<int>& feat_dims, bool pos_Set);
void get_dataSet(std::string dataSet_listFile_path, std::vector<std::string>& img_paths);
