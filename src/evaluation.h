#pragma once

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\ml\ml.hpp>

#include <iostream>

void non_Max_Suppression(std::vector<std::vector<float>>& final_BBox, std::vector<std::vector<float>> detWinFeat, int temp_Width, int temp_Height);
void detectionWindow_features(std::vector<std::vector<float>>& detWinFeat, int x, int y, float scale, float score);
void showMaximabBoxes(std::vector<std::vector<float>>& final_BBox, std::string img_Path, std::vector<int> & bBoxesOrig);
