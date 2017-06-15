#pragma once

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include <iostream>
#include <String>
#include <fstream>

void  getBoundingBox(std::string annotationList, std::vector<std::vector<int>>& boundingBoxes);

bool is_detection_Ok();

void get_HOG_feat_train(std::string img_list_file_path, double ***&featArray, const int cell_size, std::vector<int>& dims);