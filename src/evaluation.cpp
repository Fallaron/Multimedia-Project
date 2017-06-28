#include "evaluation.h"

using namespace cv;
using namespace std;

// suppress detected bBoxes (remain with a single box if no. persons is 1 else 2 if two persons etc)
//for a all scales and add results to final bBoxes to be evaluated by is_Detection_ok
void non_Max_Suppression(std::vector<std::vector<float>>& final_BBox, std::vector<std::vector<float>> detWinFeat, int temp_Width, int temp_Height) {
	final_BBox = std::vector<std::vector<float>>();
	int dims = sizeof(detWinFeat[0][0]);
	int boxes = sizeof(detWinFeat) / dims;
	int sel = 0;
	double overlap;
	bool addNewBox = true;

	//************* initial bBox values ********
	double scale = detWinFeat[0][3];
	int x1 = detWinFeat[0][1] * scale;
	int y1 = detWinFeat[0][2] * scale;

	int width = temp_Width * scale;
	int height = temp_Height * scale;

	final_BBox[0][0] = x1;
	final_BBox[0][1] = y1;
	final_BBox[0][2] = x1 + width; //x2
	final_BBox[0][3] = y1 + height;
	final_BBox[0][4] = detWinFeat[0][0]; // score in case for later

	for (int i = 1; i < boxes; i++) {
		double scale = detWinFeat[i][3];
		int pos_x = detWinFeat[i][1] * scale;
		int pos_y = detWinFeat[i][2] * scale;

		int width = temp_Width * scale; // pos_x2
		int height = temp_Height * scale;

		// recalculate final_BBOX VECTOR Sizes in case it expanded!!
		int dims = sizeof(final_BBox[0][0]);
		int sel = sizeof(final_BBox) / dims;

		cv::Rect current_bBox(pos_x, pos_y, width, height);

		// go thrugh all best Boxes in final_bBoxes and determine if there is overlap if no add current bboxe to final_bBox array.
		for (int n = 0; n < sel; n++) {
			int x1 = final_BBox[n][0];
			int y1 = final_BBox[n][1];
			int x2 = final_BBox[n][2];
			int y2 = final_BBox[n][3];

			cv::Rect stored_bBox(x1, y1, x2 - x1, y2 - y1);
			cv::Rect intersect_rect = current_bBox & stored_bBox;
			cv::Rect union_rect = current_bBox | stored_bBox;
			overlap = intersect_rect.area() / union_rect.area();
			// there is a posiblity that we cud have more bboxes that are different.. so u have to go through them and consider only if overlap
			// that implies we dont store less than threshold overlaping bboxes in final_bBoxes
			if (overlap > 0.2) {
				addNewBox = false;
				if (detWinFeat[i][0] > final_BBox[n][4]) {
					// replace up to now best box
					final_BBox[n][0] = pos_x;
					final_BBox[n][1] = pos_y;
					final_BBox[n][2] = width + pos_x;
					final_BBox[n][3] = height + pos_y;
					final_BBox[n][4] = detWinFeat[i][0];
				}// else suprress this current bboxe

			}
		}
		// add new bBox.. cud be new pedstrian thus increase the size of final bBoxes
		if (addNewBox) {
			vector<float> temp;
			temp.push_back(pos_x);
			temp.push_back(pos_y);
			temp.push_back(width + pos_x);
			temp.push_back(width + pos_y);
			temp.push_back(detWinFeat[i][0]);
			final_BBox.push_back(temp);
			temp.clear();
		}

	}
}

void detectionWindow_features(std::vector<std::vector<float>>& detWinFeat, int x, int y, float scale, float score, int img_id) {
	// we cud consider only those windows with persons..
	int w = 0;// particular detection window generated through scanning the whole image
			  //detWinFeat = std::vector<std::vector<float>>(); initialize from outside to avoid overwrites

	vector<float> temp;
	temp.push_back(score);
	temp.push_back(x);
	temp.push_back(y);
	temp.push_back(scale);
	temp.push_back(img_id);
	detWinFeat.push_back(temp);
	/**
	detWinFeat[w][0] = score;
	detWinFeat[w][1] = x;
	detWinFeat[w][2] = y;
	detWinFeat[w][3] = scale;
	detWinFeat[w][4] = img_id;
	*/
	temp.clear();
	w++;
}