#pragma once

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include <iostream>
#include <String>
#include <fstream>

void  getBoundingBox(std::string annotationList, std::vector<std::vector<int>>& boundingBoxes);

bool is_detection_Ok();

double ** get_HOG_feat_trainSet(cv::Mat img, const int cell_size, std::vector<int>& dims);