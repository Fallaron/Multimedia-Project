#include "t1.1.h"

using namespace cv;
using namespace std;

void  getBoundingBox(std::string annotationList, std::vector<std::vector<int>>& boundingBoxes) {

	string annotationPath, line, value;
	ifstream annotationlst;
	ifstream annotationTxtFile;
	int boundingBoxLineNum = 17;
	char c;
	bool store = false;
	std::vector<int> boundingBoxValues;
	boundingBoxes = std::vector<std::vector<int>>();

	if (!annotationList.empty()) {
		// read annnotation list file from path
		annotationlst.open(annotationList);
		while (!annotationlst.eof()) {
			getline(annotationlst, annotationPath);
			// open annnotation text file from path
			annotationTxtFile.open(annotationPath);

			while (!annotationTxtFile.eof())
			{
				// get to a specific line with boundingBox info from annotation text file
				int count = 0;
				while (getline(annotationTxtFile, line)) {
					if (count == boundingBoxLineNum) {
						// look for bounding box data from this specific line

						for (int j = 69; j < line.size(); ++j) {
							c = line[j];
							while (j < line.size() && isdigit(c)) {
								value.push_back(c);
								c = line[++j];
								store = true;
							}
							if (store == true) {
								boundingBoxValues.push_back(stoi(value));
								store = false;
								value.clear();
							}
						}
						// store bounding box values for a specific row or image
						boundingBoxes.push_back(boundingBoxValues);
						boundingBoxValues.clear();
					}
					count++;
				}
			}
			annotationTxtFile.close();
		}
		annotationlst.close();
	}
}
